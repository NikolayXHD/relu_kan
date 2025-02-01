from typing import Sequence, cast
import torch
import torch.nn as nn

from .relu_kan_base import LayerBase
from .relu_kan import Layer


class ResLayer(LayerBase):
    def __init__(
        self,
        input_len: int,
        input_channels: int,
        g: int,
        k: int,
        output_len: int,
        output_channels: int,
        center_zero: bool,
        final_output_len: int,
        final_output_channels: int,
    ) -> None:
        super().__init__(
            input_len=input_len,
            input_channels=input_channels,
            g=g,
            k=k,
            center_zero=center_zero,
        )

        self.output_len = output_len
        self.output_channels = output_channels

        self.final_output_len = final_output_len
        self.final_output_channels = final_output_channels

        self.fc = nn.Linear(
            in_features=self.num_spline_outputs,
            out_features=output_channels * output_len,
        )

        self.fc_final = nn.Linear(
            in_features=self.num_spline_outputs,
            out_features=final_output_channels * final_output_len,
        )

        torch.nn.init.constant_(self.fc.bias, 0)
        # kaming normal with default mode fan_in introduces another factor of
        # num_spline_outputs ** -0.5
        torch.nn.init.kaiming_normal_(
            self.fc.weight, mode='fan_in', nonlinearity='relu'
        )

        torch.nn.init.constant_(self.fc_final.bias, 0)
        # kaming normal with default mode fan_in introduces another factor of
        # num_spline_outputs ** -0.5
        torch.nn.init.kaiming_normal_(
            self.fc_final.weight, mode='fan_in', nonlinearity='relu'
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch, input_channels, input_len)
        x_splines = self._forward_splines(x)
        x_splines = x_splines.flatten(1, 3)
        # (batch, (grid + k) * input_channels * input_len)
        x = self.fc(x_splines)  # (batch, output_channels * output_len)
        x = x.reshape(-1, self.output_channels, self.output_len)

        x_final = self.fc_final(x_splines)
        x_final = x_final.reshape(
            -1, self.final_output_channels, self.final_output_len
        )
        return x, x_final


class ResReLUKAN(nn.Module):
    def __init__(
        self,
        layers_len: Sequence[int],
        layers_channels: Sequence[int],
        g: int,
        k: int,
        center_zero: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ResLayer(
                    input_len=input_len,
                    input_channels=input_channels,
                    g=g,
                    k=k,
                    output_len=output_len,
                    output_channels=output_channels,
                    center_zero=center_zero,
                    final_output_len=layers_len[-1],
                    final_output_channels=layers_channels[-1],
                )
                for (
                    input_len,
                    input_channels,
                    output_len,
                    output_channels,
                ) in zip(
                    layers_len[:-2],
                    layers_channels[:-2],
                    layers_len[1:],
                    layers_channels[1:],
                )
            ]
        )

        self.final_layer = Layer(
            input_len=layers_len[-2],
            input_channels=layers_channels[-2],
            g=g,
            k=k,
            output_len=layers_len[-1],
            output_channels=layers_channels[-1],
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
        result = torch.Tensor | None

        layer: ResLayer
        for layer in self.layers:
            x, x_final = layer(x)
            if result is None:
                result = x_final
            else:
                result += x_final

        x_final = self.final_layer(x)
        result += x_final
        return result


__all__ = ['ResReLUKAN']
