from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from .relu_kan_base import LayerBase
from .conv_2d_spec import Conv2dModelSpec, Conv2dLayerSpec, Conv2dBlockSpec


class ResConv2dBlock(nn.Module):
    skip_projection: Projection2d | None

    def __init__(self, spec: Conv2dBlockSpec) -> None:
        super().__init__()

        self.input_shape = spec.input_shape
        self.input_channels = spec.input_channels
        self.output_shape = spec.output_shape
        self.output_channels = spec.output_channels

        if spec.requires_projection:
            self.skip_projection = Projection2d(spec.first_layer_spec)
        else:
            self.skip_projection = None

        self.layers = nn.Sequential(
            Conv2dLayer(spec.first_layer_spec),
            *(
                Conv2dLayer(spec.next_layer_spec)
                for _ in range(1, spec.num_layers)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_projection is None:
            identity = x
        else:
            identity = self.skip_projection(x)

        residual = self.layers(x)
        y = identity + residual
        return y


class Conv2dLayer(LayerBase):
    batch_norm: nn.BatchNorm2d | None

    def __init__(self, spec: Conv2dLayerSpec) -> None:
        super().__init__(
            input_len=spec.input_shape[0] * spec.input_shape[1],
            input_channels=spec.input_channels,
            g=spec.kan_spec.g,
            k=spec.kan_spec.k,
            center_zero=spec.kan_spec.center_zero,
        )

        self.input_shape = spec.input_shape
        self.input_channels = spec.input_channels
        self.output_shape = spec.output_shape
        self.output_channels = spec.output_channels

        self.scale = spec.kan_spec.scale

        if spec.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=self.input_channels)
            if self.batch_norm.bias is not None:
                torch.nn.init.constant_(self.batch_norm.bias, 0)

            # zero-init residual connection
            torch.nn.init.constant_(self.batch_norm.weight, 0)
        else:
            self.batch_norm = None

        self.conv = nn.Conv2d(
            in_channels=self.num_spline_channels,
            out_channels=spec.output_channels,
            kernel_size=spec.conv_spec.kernel,
            stride=spec.conv_spec.stride,
            padding=spec.conv_spec.padding,
            groups=(
                1
                if spec.conv_spec.input_channels_interact
                else spec.kan_spec.g + spec.kan_spec.k
            ),
        )

        torch.nn.init.normal_(
            self.conv.weight,
            0,
            (
                self.num_spline_channels
                * spec.conv_spec.kernel[0]
                * spec.conv_spec.kernel[1]
            )
            ** -0.5,
        )

        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, input_channels, input_sz_y, input_sz_x)

        if self.batch_norm is None:
            x_normal = x
        else:
            x_normal = self.batch_norm(x)
        x_scaled = x_normal * self.scale

        x_splines = self._forward_splines(x_scaled.flatten(2, 3))
        # (batch, g + k, input_channels, input_sz_y * input_sz_x)

        x_splines_2d = x_splines.flatten(1, 2).reshape(
            x_splines.shape[0],
            x_splines.shape[1] * x_splines.shape[2],
            *self.input_shape,
        )
        # (batch, (g + k) * input_channels, input_sz_y, input_sz_x)

        y = self.conv(x_splines_2d)
        # (batch, output_channels, output_sz_y, output_sz_x)

        return y


class Projection2d(nn.Module):
    def __init__(self, spec: Conv2dLayerSpec) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=spec.conv_spec.kernel,
            stride=spec.conv_spec.stride,
            padding=spec.conv_spec.padding,
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=spec.conv_spec.kernel,
            stride=spec.conv_spec.stride,
            padding=spec.conv_spec.padding,
        )
        self.skip_projection = nn.Conv2d(
            in_channels=spec.input_channels * 2,  # avg / max pooling
            out_channels=spec.output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,  # channels interact
        )

        if self.skip_projection.bias is not None:
            torch.nn.init.constant_(self.skip_projection.bias, 0)

        torch.nn.init.normal_(
            self.skip_projection.weight,
            0,
            (
                spec.input_channels
                * spec.conv_spec.kernel[0]
                * spec.conv_spec.kernel[1]
            )
            ** -0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)

        # Concatenate along the channel dimension
        concatenated = torch.cat((max_pooled, avg_pooled), dim=1)
        y = self.skip_projection(concatenated)
        return y


class ResConv2dReLUKAN(nn.Module):
    output_projection: Projection2d | None

    def __init__(self, spec: Conv2dModelSpec) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *(ResConv2dBlock(spec) for spec in spec.blocks)
        )
        if spec.final_projection_spec.requires_projection:
            self.output_projection = Projection2d(
                spec.final_projection_spec.first_layer_spec
            )
        else:
            self.output_projection = None

    @property
    def mask(self) -> torch.Tensor | None:
        return cast(Conv2dLayer, cast(ResConv2dBlock, self.blocks[0]))._mask

    @mask.setter
    def mask(self, mask: torch.Tensor | None) -> None:
        cast(Conv2dLayer, cast(ResConv2dBlock, self.blocks[0]))._mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, input_channels, input_sz_y, input_sz_x)
        embeddings = self.blocks(x)
        if self.output_projection is None:
            return embeddings

        y = self.output_projection(embeddings)
        return y


__all__ = ['ResConv2dReLUKAN']
