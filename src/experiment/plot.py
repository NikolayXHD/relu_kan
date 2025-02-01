from typing import Sequence
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

from .eval_hist import EvalHist


def plot_progress(
    eval_hist: EvalHist,
    loss_name: str,
    metric_name: str,
    metric_larger_is_better: bool,
    baseline_metrics: Sequence[float],
    baseline_metric_names: Sequence[str],
    min_epochs: int = 0,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    if figsize is None:
        figsize = (6, 4)

    valid_loss = eval_hist.valid_loss.numpy()
    valid_metric = eval_hist.valid_metric.numpy()

    if metric_larger_is_better:
        best_metric = np.max(valid_metric)
    else:
        best_metric = np.min(valid_metric)

    last_metric = valid_metric[-1]

    best_loss = np.min(valid_loss)
    last_loss = valid_loss[-1]

    learning_rate = eval_hist.learning_rate.numpy()
    assert valid_loss.shape == learning_rate.shape

    if len(valid_loss) < min_epochs:
        valid_loss = np.pad(
            valid_loss,
            (0, min_epochs - len(valid_loss)),
            mode='constant',
            constant_values=np.nan,
        )
        valid_metric = np.pad(
            valid_metric,
            (0, min_epochs - len(valid_metric)),
            mode='constant',
            constant_values=np.nan,
        )
        learning_rate = np.pad(
            learning_rate,
            (0, min_epochs - len(learning_rate)),
            mode='constant',
            constant_values=np.nan,
        )

    epochs = 1 + np.arange(len(valid_loss))
    fig, (ax1, ax2, ax3) = plt.subplots(
        figsize=figsize,
        nrows=3,
        height_ratios=[5, 2, 1],
        sharex=True,
        constrained_layout=True,
    )

    ax1.plot(
        epochs,
        valid_metric,
        '.-',
        label=f'{metric_name} {last_metric:.4g} best {best_metric:.4g}',
        linewidth=1,
        markersize=3,
    )

    for baseline_metric, name in zip(baseline_metrics, baseline_metric_names):
        baseline_arr = np.full_like(
            epochs, fill_value=baseline_metric, dtype=np.float32
        )
        ax1.plot(
            epochs,
            baseline_arr,
            '-.',
            linewidth=1,
            label=f'{name} {baseline_metric:.4g}',
        )

    ax2.plot(
        epochs,
        valid_loss,
        '.-',
        label=f'{loss_name} {last_loss:.4g} best {best_loss:.4g}',
        linewidth=1,
        markersize=3,
    )

    ax3.plot(
        epochs,
        learning_rate,
        '.-',
        label='learning rate',
        linewidth=1,
        markersize=3,
    )

    for ax in (ax1, ax2, ax3):
        ax.minorticks_on()
        ax.grid(which='major', alpha=0.4)
        ax.grid(which='minor', alpha=0.1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(nbins=25, integer=True))

    ax1.legend(loc='upper right', framealpha=0.4, fontsize='small')
    ax2.legend(loc='upper right', framealpha=0.4, fontsize='small')
    ax3.legend(loc='upper left', framealpha=0.4, fontsize='small')

    return fig, (ax1, ax2, ax3)


__all__ = ['plot_progress']
