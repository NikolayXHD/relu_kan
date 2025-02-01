# %%
from dataclasses import replace
import itertools
from typing import Sequence

from infrastructure.interrupt import InterruptHandler

from experiment import experiment_relu_kan, experiment_spec
from experiment.kan import conv_2d_relu_kan
from experiment.kan.conv_2d_spec import KanSpec

data_version = 'v2'
data_source = experiment_relu_kan.version_to_data_source[data_version]
print(
    data_version,
    'train:',
    data_source.get_fold_size('train'),
    'valid:',
    data_source.get_fold_size('test'),
)


def train_func(
    n: int | Sequence[int],
    num_channels_base: int,
    end_epoch: int | None = None,
) -> None:
    num_block_repeats: Sequence[int]
    if isinstance(n, int):
        assert n > 0
        num_block_repeats = [1 + n, n, n, n]
        # num_layers = num_block_repeats x blocks_num_layers
        # in ResNet paper:  [1 + 2n, 2n, 2n, 2n], n=3, 5, 9
        # here is the same: [1 + 2n, 2n, 2n, 2n]
    else:
        assert all(isinstance(ni, int) and ni > 0 for ni in n)
        num_block_repeats = n

    first_block_downscale = False
    first_block_expand = True

    num_block_types = len(num_block_repeats)
    assert num_block_types > 0
    assert all(repeats > 0 for repeats in num_block_repeats)

    args = replace(
        experiment_relu_kan.TrainArgs(
            data_version=data_version,
            augment=True,  # like in ResNet paper
            batch_len=32,  # 128 in ResNet paper
            end_epoch=end_epoch,
            lr=1e-4,  # 0.1 in ResNet paper
            weight_decay=1e-2,  # 0.0001 in ResNet paper
            weight_decay_splines=False,
            model_spec=experiment_spec.Conv2dModelSpecRepr(
                input_shape=data_source.input_shape,
                input_channels=data_source.num_channels,
                output_channels=data_source.num_classes,
                blocks_kernel=[(3, 3)] * sum(num_block_repeats),
                blocks_stride=list(
                    itertools.chain(
                        *(
                            (
                                [(1, 1)] * repeats
                                if i == 0 and not first_block_downscale
                                else [(2, 2)] + [(1, 1)] * (repeats - 1)
                            )
                            for i, repeats in enumerate(num_block_repeats)
                        )
                    )
                ),
                blocks_padding=[(1, 1)] * sum(num_block_repeats),
                blocks_channels=[
                    num_channels_base
                    * 2 ** (i if first_block_expand else max(0, i - 1))
                    for i, repeats in enumerate(num_block_repeats)
                    for _ in range(repeats)
                ],
                blocks_channels_interact=[True] * sum(num_block_repeats),
                blocks_num_layers=[2] * sum(num_block_repeats),
                blocks_use_batch_norm=[True] * sum(num_block_repeats),
                kan_spec=KanSpec(g=5, k=3, center_zero=True, scale=0.375),
            ),
            model_cls=conv_2d_relu_kan.ResConv2dReLUKAN,
            # train_from_scratch=True,
        )
    )

    with InterruptHandler() as interrupt_handler:
        experiment_relu_kan.train(
            args, interrupt_handler=interrupt_handler, seed=42
        )


# %%
train_func((4, 1, 1, 1), 16, end_epoch=61)

# %%
train_func((4, 1, 1, 1), 32, end_epoch=32)

# %%
train_func((4, 1, 1, 1), 64, end_epoch=72)

# %%
train_func((4,), 64, end_epoch=82)
