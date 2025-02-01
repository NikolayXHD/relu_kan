import numpy as np
from PIL.PngImagePlugin import PngImageFile
import torch

from . import data_source
from .data_source_v1 import DataSourceV1


class DataSourceV2(DataSourceV1):
    def _define_input_num_channels(self) -> int:
        # hue, saturation, value
        # where hue is split into 2 trigonometric channels to reflect its
        # circular nature
        return 3

    def _build_tensor(self, img: PngImageFile) -> torch.Tensor:
        img_hsv = img.convert('RGB')
        r = torch.Tensor(np.array(img_hsv.getchannel('R')))
        g = torch.Tensor(np.array(img_hsv.getchannel('G')))
        b = torch.Tensor(np.array(img_hsv.getchannel('B')))

        assert r.shape == g.shape == b.shape == self.input_shape

        # [-0.5 .. 0.5]
        sample_feature = torch.stack([r, g, b], dim=0) / 255.0 - 0.5
        assert sample_feature.shape == (self.num_channels, *self.input_shape)
        return sample_feature


__all__ = ['DataSourceV2']
