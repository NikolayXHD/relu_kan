from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def load_eval_hist(path: Path) -> EvalHist:
    hist = torch.load(path, weights_only=True)

    assert isinstance(hist['valid_loss'], torch.Tensor)
    assert hist['valid_loss'].ndim == 1

    assert isinstance(hist['valid_metric'], torch.Tensor)
    assert hist['valid_metric'].ndim == 1

    assert isinstance(hist['learning_rate'], torch.Tensor)
    assert hist['learning_rate'].ndim == 1

    return EvalHist(
        valid_loss=hist['valid_loss'],
        valid_metric=hist['valid_metric'],
        learning_rate=hist['learning_rate'],
    )


def save_eval_hist(hist: EvalHist, path: Path) -> None:
    torch.save(hist.state_dict(), path)


@dataclass
class EvalHist:
    valid_loss: torch.Tensor
    valid_metric: torch.Tensor
    learning_rate: torch.Tensor

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            'valid_loss': self.valid_loss,
            'valid_metric': self.valid_metric,
            'learning_rate': self.learning_rate,
        }


__all__ = ['load_eval_hist', 'save_eval_hist', 'EvalHist']
