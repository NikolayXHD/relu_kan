from dataclasses import replace
from typing import Callable

from IPython.display import clear_output
from matplotlib import pyplot as plt
import shutil
from experiment.eval_hist import EvalHist
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from infrastructure.interrupt import InterruptHandler

from . import experiment_base, plot, data_source_v1, data_source_v2
from .kan import conv_2d_relu_kan
from .experiment_spec import TrainArgs, PlotArgs, dataset_dir

# https://en.wikipedia.org/wiki/CIFAR-10
# https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
# ResNet 20 0.27M 8.75
# ResNet 32 0.46M 7.51
# ResNet 44 0.66M 7.17
# ResNet 56 0.85M 6.97
# ResNet 110 1.7M 6.43 (6.61±0.16)
baseline_errate_resnet_20 = 8.75 / 100
baseline_errate_resnet_110 = 6.97 / 100

version_to_data_source = {
    'v1': data_source_v1.DataSourceV1(cache_dir=dataset_dir('v1')),
    'v2': data_source_v2.DataSourceV2(cache_dir=dataset_dir('v2')),
}


def train(
    args: TrainArgs,
    plot_args: PlotArgs = PlotArgs(),
    base_model_args: TrainArgs | None = None,
    plot_func: Callable[[TrainArgs, PlotArgs], None] | None = None,
    seed: int | None = None,
    interrupt_handler: InterruptHandler | None = None,
) -> torch.nn.Module:
    if plot_func is None:
        plot_func = plot_progress

    data_source = version_to_data_source[args.data_version]

    last_epoch = experiment_base.last_epoch(args.experiment_dir)
    if last_epoch is None:
        last_epoch = -1
    if args.end_epoch is not None:
        args = replace(args, num_epochs=args.end_epoch - last_epoch - 1)

    if args.train_from_scratch is True:
        if last_epoch < 20 or input(
            f'Delete train progress of {last_epoch + 1} epochs? [y/n]'
        ).startswith('y'):
            shutil.rmtree(args.experiment_dir, ignore_errors=True)

    assert args.model_cls is conv_2d_relu_kan.ResConv2dReLUKAN
    model = conv_2d_relu_kan.ResConv2dReLUKAN(spec=args.model_spec)

    if args.weight_decay_splines:
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        params_to_decay = []
        params_to_not_decay = []

        for name, param in model.named_parameters():
            if (
                name.endswith('.phase_low')
                or name.endswith('.phase_high')
                or name.endswith('.mu')
                or name.endswith('.sigma')
            ):
                params_to_not_decay.append(param)
            else:
                params_to_decay.append(param)

        optimizer = AdamW(
            [
                {'params': params_to_decay, 'weight_decay': args.weight_decay},
                {'params': params_to_not_decay, 'weight_decay': 0},
            ],
            lr=args.lr,
        )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=2)

    def _plot() -> None:
        if plot_func is None:
            return
        clear_output()
        plot_func(args, plot_args)
        plt.show()

    _plot()

    assert args.num_epochs is not None

    if base_model_args is not None:
        # Basically only learning rate controls are allowed to change,
        # such as args.lr, args.batch_size, args.weight_decay, and so on.
        assert base_model_args.data_version == args.data_version
        assert base_model_args.model_cls == args.model_cls
        assert base_model_args.model_spec == args.model_spec



    experiment_base.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        experiment_dir=args.experiment_dir,
        dataset_dir=args.dataset_dir,
        data_src=data_source,
        feature_rescale=None,
        batch_size=args.batch_len,
        augment=args.augment,
        seed=seed,
        end_epoch_callback=_plot,
        interrupt_handler=interrupt_handler,
    )

    return model


def plot_progress(args: TrainArgs, plot_args: PlotArgs) -> None:
    eval_hist = experiment_base.maybe_load_eval_hist(args.experiment_dir)
    if eval_hist is None:
        return

    eval_hist_err_rate = EvalHist(
        valid_loss=eval_hist.valid_loss,
        learning_rate=eval_hist.learning_rate,
        valid_metric=1 - eval_hist.valid_metric,
    )

    fig, (ax1, ax2, ax3) = plot.plot_progress(
        eval_hist=eval_hist_err_rate,
        min_epochs=plot_args.min_epochs,
        figsize=plot_args.figsize,
        loss_name='cross-entropy',
        metric_name='Error rate',
        metric_larger_is_better=False,
        baseline_metrics=(
            baseline_errate_resnet_20,
            baseline_errate_resnet_110,
        ),
        baseline_metric_names=(
            'Error rate ResNet 20',
            'Error rate ResNet 110',
        ),
    )
    ax1.set_ylim(0, None)
    ax3.set_ylim(0, None)


__all__ = ['train', 'plot_progress', 'PlotArgs', 'TrainArgs']
