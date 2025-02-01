from dataclasses import dataclass
from functools import cached_property
from hashlib import sha1
from pathlib import Path
from typing import Sequence, Type


from experiment.kan.conv_2d_spec import Conv2dModelSpec
from file_sys.directories import REPO_DIR

from .kan import conv_2d_relu_kan


@dataclass
class Conv2dModelSpecRepr(Conv2dModelSpec):
    def __str__(self) -> str:
        hash_input = (
            f'_g{self.kan_spec.g}'
            f'_k{self.kan_spec.k}'
            f'_cz{int(self.kan_spec.center_zero)}'
            f'_s{self.kan_spec.scale}'
            f'_is{_repr_int_2d_tuples([self.input_shape])}'
            f'_ic{self.input_channels}'
            f'_oc{self.output_channels}'
            f'_k{_repr_int_2d_tuples(self.blocks_kernel)}'
            f'_s{_repr_int_2d_tuples(self.blocks_stride)}'
            f'_p{_repr_int_2d_tuples(self.blocks_padding)}'
            f'_c{_repr_ints(self.blocks_channels)}'
            f'_b{_repr_ints(list(map(int, self.blocks_use_batch_norm)))}'
            f'_i{_repr_ints(list(map(int, self.blocks_channels_interact)))}'
            f'_n{_repr_ints(self.blocks_num_layers)}'
        )
        # hashed = _sha1_hash(hash_input)
        return f'res_conv_2d_relu_kan_{hash_input}'


@dataclass
class TrainArgs:
    data_version: str
    augment: bool
    lr: float
    weight_decay: float
    model_spec: Conv2dModelSpecRepr
    model_cls: Type[conv_2d_relu_kan.ResConv2dReLUKAN] = (
        conv_2d_relu_kan.ResConv2dReLUKAN
    )
    weight_decay_splines: bool = True
    t0: int = 10
    batch_len: int = 512
    path_discriminator: str = ''
    num_epochs: int | None = None
    end_epoch: int | None = None
    train_from_scratch: bool | None = None

    def __post_init__(self) -> None:
        if self.num_epochs is None:
            self.num_epochs = self.t0 * (1 + 2 + 4 + 8)

    @cached_property
    def dataset_dir(self) -> Path:
        return dataset_dir(self.data_version)

    @cached_property
    def experiment_dir(self) -> Path:
        assert self.model_cls is conv_2d_relu_kan.ResConv2dReLUKAN
        assert self.model_spec is not None
        result = (
            self.dataset_dir
            / f'augm_{int(self.augment)}'
            / str(self.model_spec)
            / (
                f'wd_{self.weight_decay:g}'
                f'{'_no_wd_spl' if not self.weight_decay_splines else ''}'
                f'_lr_{self.lr:g}_b{self.batch_len}{self.path_discriminator}'
            )
        )
        return result


@dataclass(frozen=True)
class PlotArgs:
    figsize: tuple[float, float] | None = None
    min_epochs: int = 30


def _repr_int_2d_tuples(v: Sequence[tuple[int, int]]) -> str:
    separate = any(v1 > 9 or v2 > 9 for v1, v2 in v)
    result = ''
    for v1, v2 in v:
        if separate and result != '':
            result += '-'
        if v1 == v2:
            result += str(v1)
        elif separate:
            result += f'{v1}x{v2}'
        else:
            result += f'{v1}{v2}'
    return result


def _repr_ints(v: Sequence[int]) -> str:
    separate = any(v1 > 9 for v1 in v)
    if separate:
        return '-'.join(map(str, v))
    else:
        return ''.join(map(str, v))


def _sha1_hash(value: str) -> str:
    """hash string of 40 hex characters"""
    h = sha1()
    h.update(value.encode('utf-8'))
    return h.hexdigest()


def dataset_dir(data_version: str) -> Path:
    return REPO_DIR / 'experiments.large' / data_version


__all__ = [
    'Conv2dModelSpecRepr',
    '_repr_int_2d_tuples',
    '_repr_ints',
    '_sha1_hash',
    'dataset_dir',
]
