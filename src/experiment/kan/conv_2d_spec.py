from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence


@dataclass
class Conv2dModelSpec:
    input_shape: tuple[int, int]
    input_channels: int
    output_channels: int
    blocks_kernel: Sequence[tuple[int, int]]
    blocks_stride: Sequence[tuple[int, int]]
    blocks_padding: Sequence[tuple[int, int]]
    blocks_channels: Sequence[int]
    blocks_channels_interact: Sequence[bool]
    blocks_num_layers: Sequence[int]
    blocks_use_batch_norm: Sequence[bool]
    kan_spec: KanSpec
    blocks: Sequence[Conv2dBlockSpec] = field(init=False)
    final_projection: Conv2dBlockSpec = field(init=False)

    def __post_init__(self) -> None:
        blocks: list[Conv2dBlockSpec] = []

        input_shape = self.input_shape
        input_channels = self.input_channels
        for (
            kernel,
            stride,
            padding,
            channels,
            channels_interact,
            num_conv,
            use_batch_norm,
        ) in zip(
            self.blocks_kernel,
            self.blocks_stride,
            self.blocks_padding,
            self.blocks_channels,
            self.blocks_channels_interact,
            self.blocks_num_layers,
            self.blocks_use_batch_norm,
        ):
            assert all(isinstance(k, int) and k > 0 for k in kernel)
            assert all(isinstance(s, int) and s > 0 for s in stride)
            assert all(isinstance(p, int) and p >= 0 for p in padding)
            assert isinstance(channels, int) and channels > 0
            assert isinstance(num_conv, int) and num_conv > 0

            block = Conv2dBlockSpec(
                input_shape=input_shape,
                input_channels=input_channels,
                output_channels=channels,
                conv_spec=Conv2dSpec(
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    input_channels_interact=channels_interact,
                ),
                kan_spec=self.kan_spec,
                num_layers=num_conv,
                use_batch_norm=use_batch_norm,
            )
            blocks.append(block)
            input_shape = block.output_shape
            input_channels = block.output_channels
        self.blocks = blocks

        last_block = blocks[-1]
        self.final_projection_spec = Conv2dBlockSpec(
            input_shape=last_block.output_shape,
            input_channels=last_block.output_channels,
            output_channels=self.output_channels,
            conv_spec=Conv2dSpec(
                kernel=last_block.output_shape,
                stride=(1, 1),
                padding=(0, 0),
                input_channels_interact=True,
            ),
            kan_spec=self.kan_spec,
            num_layers=1,
            use_batch_norm=False,
        )


@dataclass
class Conv2dBlockSpec:
    input_shape: tuple[int, int]
    input_channels: int
    output_channels: int
    conv_spec: Conv2dSpec
    kan_spec: KanSpec
    num_layers: int
    use_batch_norm: bool
    output_shape: tuple[int, int] = field(init=False)
    first_layer_spec: Conv2dLayerSpec = field(init=False)
    next_layer_spec: Conv2dLayerSpec = field(init=False)
    requires_projection: bool = field(init=False)

    def __post_init__(self) -> None:
        self.output_shape = _compute_output_shape(self)
        self.first_layer_spec = Conv2dLayerSpec(
            input_shape=self.input_shape,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            conv_spec=self.conv_spec,
            kan_spec=self.kan_spec,
            use_batch_norm=self.use_batch_norm,
        )
        self.next_layer_spec = Conv2dLayerSpec(
            input_shape=self.output_shape,
            input_channels=self.output_channels,
            output_channels=self.output_channels,
            conv_spec=replace(self.conv_spec, stride=(1, 1)),
            kan_spec=self.kan_spec,
            use_batch_norm=self.use_batch_norm,
        )
        self.requires_projection = (
            self.input_channels != self.output_channels
            or self.input_shape != self.output_shape
        )


@dataclass
class Conv2dLayerSpec:
    input_shape: tuple[int, int]
    input_channels: int
    output_channels: int
    conv_spec: Conv2dSpec
    kan_spec: KanSpec
    use_batch_norm: bool
    output_shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.output_shape = _compute_output_shape(self)


@dataclass
class Conv2dSpec:
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    input_channels_interact: bool


@dataclass
class KanSpec:
    g: int
    k: int
    center_zero: bool
    scale: float


def _compute_output_shape(
    spec: Conv2dBlockSpec | Conv2dLayerSpec,
) -> tuple[int, int]:
    result = tuple(
        1 + (input_sz + 2 * padding_sz - kernel_sz) // stride_sz
        for input_sz, kernel_sz, stride_sz, padding_sz in zip(
            spec.input_shape,
            spec.conv_spec.kernel,
            spec.conv_spec.stride,
            spec.conv_spec.padding,
        )
    )
    assert len(result) == 2
    return result


__all__ = [
    'Conv2dSpec',
    'KanSpec',
    'Conv2dBlockSpec',
    'Conv2dLayerSpec',
    'Conv2dModelSpec',
]
