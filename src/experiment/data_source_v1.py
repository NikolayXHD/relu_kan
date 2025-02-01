import numpy as np
from PIL.PngImagePlugin import PngImageFile
import torch

from . import data_source

label_to_text = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# /=256 because 255 must not map to same value as 0
hue_cicle_scale = 2 * torch.pi / 256.0


class DataSourceV1(data_source.DataSourceBase):
    def _define_num_classes(self) -> int:
        return len(label_to_text)

    def _define_input_shape(self) -> tuple[int, int]:
        return (32, 32)

    def _define_input_num_channels(self) -> int:
        # hue, saturation, value
        # where hue is split into 2 trigonometric channels to reflect its
        # circular nature
        return 2 + 1 + 1

    def _build_tensors(self, fold: str) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_folds = data_source.load_huggingface_dataset('uoft-cs/cifar10')
        dataset = dataset_folds[fold]
        feature = torch.zeros(
            (len(dataset), self.num_channels, *self.input_shape),
            dtype=torch.float32,
        )
        target_raw = torch.tensor(dataset['label'], dtype=torch.long)
        target = torch.nn.functional.one_hot(
            target_raw, num_classes=self.num_classes
        ).to(torch.float32)

        img: PngImageFile
        for i, sample in enumerate(dataset):
            img = cast(PngImageFile, ['img'])
            feature[i] = self._build_tensor(img)

        return feature, target

    def _build_tensor(self, img: PngImageFile) -> torch.Tensor:
        img_hsv = img.convert('HSV')
        h = torch.Tensor(np.array(img_hsv.getchannel('H')))
        s = torch.Tensor(np.array(img_hsv.getchannel('S')))
        v = torch.Tensor(np.array(img_hsv.getchannel('V')))

        assert h.shape == s.shape == v.shape == self.input_shape

        # [-0.5 .. 0.5]
        h_cos = 0.5 * torch.cos(h * hue_cicle_scale)
        h_sin = 0.5 * torch.sin(h * hue_cicle_scale)
        s_scaled = s / 255.0 - 0.5
        v_scaled = v / 255.0 - 0.5

        sample_feature = torch.stack([h_cos, h_sin, s_scaled, v_scaled], dim=0)
        assert sample_feature.shape == (
            self.num_channels,
            *self.input_shape,
        )
        return sample_feature

    def _augment_feature(self, features: torch.Tensor) -> torch.Tensor:
        (num_samples, num_channels, sz_y, sz_x) = features.shape
        padding = 4
        features_padded = torch.nn.functional.pad(
            features, (padding,) * 4, mode='replicate'
        )

        sample_ixs = torch.arange(num_samples).view(-1, 1, 1, 1)
        # (num_samples, 1, 1, 1)
        channel_ixs = torch.arange(num_channels).view(1, -1, 1, 1)
        # (1, num_channels, 1, 1)

        x = torch.randint(0, 2 * padding, (num_samples, 1, 1, 1))
        # (num_samples, 1, 1, 1)

        y = torch.randint(0, 2 * padding, (num_samples, 1, 1, 1))
        # (num_samples, 1, 1, 1)

        x_orig = torch.arange(sz_x).view(1, 1, 1, -1)
        # (1, 1, 1, sz_x)

        y_orig = torch.arange(sz_y).view(1, 1, -1, 1)
        # (1, 1, sz_y, 1)

        x_shifted = x_orig + x
        # (num_samples, 1, 1, sz_x)

        y_shifted = y_orig + y
        # (num_samples, 1, sz_y, 1)

        features_cropped = features_padded[
            sample_ixs, channel_ixs, y_shifted, x_shifted
        ]
        # (num_samples, num_channels, sz_y, sz_x)

        flip_mask = torch.bernoulli(torch.full((num_samples,), 0.5)).bool()
        features_cropped[flip_mask] = torch.flip(
            features_cropped[flip_mask], dims=(3,)
        )

        return features_cropped


__all__ = ['DataSourceV1']
