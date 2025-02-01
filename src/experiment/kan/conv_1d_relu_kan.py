from typing import Sequence, cast
import torch
import torch.nn as nn

from .relu_kan_base import LayerBase
from .relu_kan import Layer


class ResConv1dLayer(LayerBase):
    def __init__(
        self,
        input_len: int,
        input_channels: int,
        g: int,
        k: int,
        center_zero: bool,
        final_output_len: int,
        final_output_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        input_channels_interact: bool,
        output_channels: int,
    ) -> None:
        super().__init__(
            input_len=input_len,
            input_channels=input_channels,
            g=g,
            k=k,
            center_zero=center_zero,
        )

        self.output_len = 1 + (input_len + 2 * padding - kernel_size) // stride
        self.output_channels = output_channels

        self.final_output_len = final_output_len
        self.final_output_channels = final_output_channels

        self.conv = nn.Conv1d(
            in_channels=self.num_spline_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1 if input_channels_interact else g + k,
        )

        self.fc_final = nn.Linear(
            in_features=self.num_spline_outputs,
            out_features=final_output_channels * final_output_len,
            bias=False,  # keep 1 bias on the final layer
        )

        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)
        # kaming normal with default mode fan_in introduces another factor of
        # num_spline_outputs ** -0.5

        torch.nn.init.normal_(
            self.conv.weight,
            0,
            (self.num_spline_channels * kernel_size) ** -0.5,
        )

        # torch.nn.init.kaiming_normal_(
        #     self.conv.weight, mode='fan_in', nonlinearity='relu'
        # )

        # kaming normal with default mode fan_in introduces another factor of
        # num_spline_outputs ** -0.5
        torch.nn.init.normal_(
            self.fc_final.weight,
            0,
            1e-2 * self.num_spline_outputs**-0.5,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch, input_channels, input_len)

        x_splines = self._forward_splines(x)
        # (batch, grid + k, input_channels, input_len)

        x_splines_1d = x_splines.flatten(1, 2)
        # (batch, (grid + k) * input_channels, input_len)

        x_conv = self.conv(
            x_splines_1d
        )  # (batch, output_channels, output_len)

        # (batch, (grid + k) * input_channels * input_len)
        x_splines_0d = x_splines.flatten(1, 3)
        x_final = self.fc_final(x_splines_0d)
        x_final = x_final.reshape(
            -1, self.final_output_channels, self.final_output_len
        )
        return x_conv, x_final


class ResConv1dReLUKAN(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_channels: int,
        output_len: int,
        output_channels: int,
        layers_kernel: Sequence[int],
        layers_stride: Sequence[int],
        layers_padding: Sequence[int],
        layers_channels: Sequence[int],
        layers_channels_interact: Sequence[bool],
        g: int,
        k: int,
        center_zero: bool = False,
    ) -> None:
        super().__init__()

        self.input_len = input_len
        self.input_channels = input_channels

        layers: list[ResConv1dLayer] = []

        for kernel, stride, padding, channels, channels_interact in zip(
            layers_kernel,
            layers_stride,
            layers_padding,
            layers_channels,
            layers_channels_interact,
        ):
            assert kernel > 0
            assert stride > 0
            assert padding >= 0
            assert channels > 0

            layer = ResConv1dLayer(
                input_len=input_len,
                input_channels=input_channels,
                g=g,
                k=k,
                center_zero=center_zero,
                final_output_len=output_len,
                final_output_channels=output_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                input_channels_interact=channels_interact,
                output_channels=channels,
            )

            layers.append(layer)

            input_len = layer.output_len
            input_channels = layer.output_channels

        self.layers = nn.ModuleList(layers)

        self.final_layer = Layer(
            input_len=input_len,
            input_channels=input_channels,
            g=g,
            k=k,
            output_len=output_len,
            output_channels=output_channels,
            center_zero=center_zero,
        )

    @property
    def mask(self) -> torch.Tensor | None:
        if len(self.layers) == 0:
            return self.final_layer.mask
        else:
            return cast(Layer, self.layers[0]).mask

    @mask.setter
    def mask(self, mask: torch.Tensor | None) -> None:
        if len(self.layers) == 0:
            self.final_layer.mask = mask
        else:
            cast(Layer, self.layers[0]).mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, input_channels, input_len)
        result: torch.Tensor | None = None

        layer: ResConv1dLayer
        for layer in self.layers:
            x, x_final = layer(x)
            if result is None:
                result = x_final
            else:
                result = result + x_final

        x_final = self.final_layer(x)
        result = result + x_final
        return result


__all__ = ['ResConv1dReLUKAN']
