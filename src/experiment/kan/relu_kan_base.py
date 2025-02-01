import torch
import torch.nn as nn


class LayerBase(nn.Module):
    _mask: torch.Tensor | None

    def __init__(
        self,
        input_len: int,
        input_channels: int,
        g: int,
        k: int,
        center_zero: bool,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.input_channels = input_channels

        if center_zero:
            # mean = 0.5 *(-k + (g + k)) / g
            # = 0.5
            shift = -0.5
        else:
            shift = 0

        #  -k .. g-1
        phase_low_1d = torch.arange(-k, g) / g
        # 1 .. g+k
        phase_high_1d = phase_low_1d + (k + 1) / g

        mu_1d = 0.5 * (phase_low_1d + phase_high_1d)
        sigma_1d = 0.5 * (phase_high_1d - phase_low_1d)

        mu = mu_1d.reshape(-1, 1, 1).repeat(1, input_channels, input_len)
        sigma = sigma_1d.reshape(-1, 1, 1).repeat(1, input_channels, input_len)

        self.mu = nn.Parameter(mu + shift)
        self.sigma = nn.Parameter(sigma)
        # (grid + k, input_channels, input_len)

        self._mask = None
        self.num_spline_channels = (g + k) * input_channels
        self.num_spline_outputs = self.num_spline_channels * input_len

        spline_max = ((k + 1) / (2 * g)) ** 4
        self.output_scale = spline_max**-1

    @property
    def mask(self) -> torch.Tensor | None:
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor | None) -> None:
        if mask is not None:
            assert mask.shape == (self.input_channels, self.input_len)
            assert mask.dtype == torch.bool
        self._mask = mask

    def _forward_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Output shape: (batch, g + k, input_channels, input_len)"""
        assert x.ndim == 3
        # (batch, input_channels, input_len)

        x = x.unsqueeze(1)
        # (batch, 1, input_channels, input_len)

        # softmax

        # x = x - self.mu
        # # (batch, g + k, input_channels, input_len)

        # x = nn.functional.log_softmax(-(x**2), dim=1).exp()
        # # (batch, g + k, input_channels, input_len)

        # normal cdf manual

        # x = torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (
        #     self.sigma * (2 * torch.pi) ** 2
        # )

        # normal cdf

        # x = x - self.mu
        # (batch, g + k, input_channels, input_len)

        # x = torch.distributions.Normal(0, self.sigma.abs()).log_prob(x).exp()

        # relu

        # x = self.output_scale * (
        #     torch.relu((x - self.phase_low) * (self.phase_high - x)) ** 2
        # )  # (batch, g + k, input_channels, input_len)

        # decoupled

        x = (x - self.mu) / self.sigma
        x = torch.relu(1 - x**2) ** 2
        # (batch, g + k, input_channels, input_len)

        if self._mask is not None:
            x = x * self._mask

        return x


__all__ = ['LayerBase']
