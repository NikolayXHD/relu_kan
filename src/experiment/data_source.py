from __future__ import annotations

from abc import abstractmethod, ABC
from functools import cache
from pathlib import Path
from typing import Iterable

from datasets import DatasetDict, load_dataset
from tqdm.notebook import tqdm
from safetensors.torch import load_file, save_file
import torch


class DataSourceBase(ABC):
    def __init__(self, cache_dir: Path) -> None:
        self.fold_to_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.input_shape = self._define_input_shape()
        self.num_channels = self._define_input_num_channels()
        self.num_classes = self._define_num_classes()
        self.cache_dir = cache_dir

    @abstractmethod
    def _build_tensors(
        self, fold: str
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def _define_input_shape(self) -> tuple[int, int]: ...

    @abstractmethod
    def _define_input_num_channels(self) -> int: ...

    @abstractmethod
    def _define_num_classes(self) -> int: ...

    def get_tensors(self, fold: str) -> tuple[torch.Tensor, torch.Tensor]:
        if fold not in self.fold_to_tensors:
            cache_path = self.cache_dir / f'{fold}.safetensors'
            if cache_path.exists():
                body = load_file(cache_path)
                x: torch.Tensor = body['x']
                y: torch.Tensor = body['y']
            else:
                x, y = self._build_tensors(fold)
                body = {'x': x, 'y': y}
                self.cache_dir.mkdir(exist_ok=True, parents=True)
                save_file(body, cache_path)
            self.fold_to_tensors[fold] = x, y
        return self.fold_to_tensors[fold]

    def get_fold_size(self, fold: str) -> int:
        return self.get_tensors(fold)[0].shape[0]

    def iter_batches(
        self,
        fold: str,
        batch_size: int,
        device: torch.device,
        augment: bool,
        tqdm_desc: str | None = None,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over the dataset in batches.

        Batches are represented as 2 tensors: feature, target.

        feature is 4d tensor with shape
        (batch_size, num_channels, input_size_x, input_size_y)

        target is 2d tensors with shape (batch_size, num_classes)
        """
        feature_raw, target_raw = self.get_tensors(fold)
        assert feature_raw.ndim == 4
        assert target_raw.ndim == 2
        assert feature_raw.shape[0] == target_raw.shape[0]

        if augment:
            feature = self._augment_feature(feature_raw).to(device)
        else:
            feature = feature_raw.to(device)
        # (num_samples, num_channels, sz_x, sz_y)

        target = target_raw.to(device)
        # (num_samples, num_classes)

        ix = torch.arange(len(target), device=device)

        for ix_batch in tqdm(
            torch.split(ix, batch_size),
            desc=tqdm_desc,
            leave=False,
            mininterval=1.0,
        ):
            feature_batch = feature[ix_batch]
            # (batch_size, num_channels, sz_x, sz_y)

            target_batch = target[ix_batch]
            # (batch_size, num_classes)

            yield feature_batch, target_batch

    def _augment_feature(self, features: torch.Tensor) -> torch.Tensor:
        return features


@cache
def load_huggingface_dataset(path: str) -> DatasetDict:
    ds = load_dataset(path)
    assert isinstance(ds, DatasetDict)
    return ds


__all__ = ['DataSourceBase']
