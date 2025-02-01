from __future__ import annotations

from dataclasses import dataclass
from logging import warning
from pathlib import Path
from typing import Callable

import delu
import torch
import torchmetrics

from infrastructure.interrupt import InterruptHandler

from . import eval_hist, data_source

DEVICE_CPU = torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else DEVICE_CPU


@dataclass
class CheckpointFeatures:
    epoch: int


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    experiment_dir: Path,
    dataset_dir: Path,
    data_src: data_source.DataSourceBase,
    feature_rescale: tuple[float, float] | None,
    batch_size: int,
    augment: bool,
    seed: int | None = None,
    end_epoch_callback: Callable[[], None] | None = None,
    interrupt_handler: InterruptHandler | None = None,
) -> None:
    if interrupt_handler is not None and interrupt_handler.interrupt_requested:
        return

    experiment_dir.mkdir(exist_ok=True, parents=True)
    dataset_dir.mkdir(exist_ok=True, parents=True)

    valid_loss: list[float]
    valid_metric: list[float]
    learning_rates: list[float]
    model = model.to(DEVICE)

    checkpoint_features = maybe_load_last_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_dir=experiment_dir,
    )
    if checkpoint_features is not None:
        epoch_start = 1 + checkpoint_features.epoch
        eval_hist_ = maybe_load_eval_hist(experiment_dir)
        assert eval_hist_ is not None
    else:
        epoch_start = 0
        if seed is not None:
            delu.random.seed(seed)
        eval_hist_ = eval_hist.EvalHist(
            valid_loss=torch.zeros(0),
            valid_metric=torch.zeros(0),
            learning_rate=torch.zeros(0),
        )

    valid_loss = eval_hist_.valid_loss.tolist()
    valid_metric = eval_hist_.valid_metric.tolist()
    learning_rates = eval_hist_.learning_rate.tolist()
    # support scenario where we remove some last checkpoints
    # to resume from the one we like
    valid_loss = valid_loss[:epoch_start]
    learning_rates = learning_rates[:epoch_start]

    assert (
        len(valid_loss) == len(valid_metric) == epoch_start
    ), f'{len(valid_loss)=} {len(valid_metric)=} {epoch_start=}'

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    metric_func = torchmetrics.Accuracy(
        task='multiclass',
        num_classes=data_src.num_classes,
        average='weighted',
        multidim_average='global',
    ).to(DEVICE)

    for epoch in range(epoch_start, epoch_start + num_epochs):
        if (
            interrupt_handler is not None
            and interrupt_handler.interrupt_requested
        ):
            return
        for feature, target in data_src.iter_batches(
            'train',
            batch_size=batch_size,
            device=DEVICE,
            augment=augment,
            tqdm_desc=f'train epoch {epoch}',
        ):
            if (
                interrupt_handler is not None
                and interrupt_handler.interrupt_requested
            ):
                return

            if feature_rescale is not None:
                feature = feature * feature_rescale[1] + feature_rescale[0]

            # assert torch.isnan(feature).sum() == 0
            optimizer.zero_grad()
            logits_raw = model(feature)
            # (batch_size, num_classes, 1, 1)

            logits = logits_raw.flatten(start_dim=-3, end_dim=-1)
            loss = torch.mean(loss_func(logits, target))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            valid_total_loss = 0.0
            valid_total_metric = 0.0
            valid_count = 0.0

            for feature, target in data_src.iter_batches(
                'test',
                batch_size=batch_size,
                device=DEVICE,
                augment=False,
                tqdm_desc=f'eval epoch {epoch}',
            ):
                if feature_rescale is not None:
                    feature = feature * feature_rescale[1] + feature_rescale[0]

                # assert torch.isnan(feature).sum() == 0
                logits_raw = model(feature)
                # (batch_size, num_classes, 1, 1)

                logits = logits_raw.flatten(start_dim=-3, end_dim=-1)
                loss = loss_func(logits, target)

                target_cat = torch.argmax(target, dim=1)
                metric = metric_func(logits, target_cat)  # scalar
                valid_total_metric += float(metric * target.shape[0])
                valid_total_loss += float(torch.sum(loss))
                valid_count += target.shape[0]

        valid_loss_mean = valid_total_loss / valid_count
        valid_metric_mean = valid_total_metric / valid_count

        valid_loss.append(valid_loss_mean)
        valid_metric.append(valid_metric_mean)
        learning_rates.append(scheduler.get_last_lr()[0])

        eval_hist_.valid_loss = torch.tensor(valid_loss)
        eval_hist_.valid_metric = torch.tensor(valid_metric)
        eval_hist_.learning_rate = torch.tensor(learning_rates)

        scheduler.step()

        save_eval_hist(eval_hist_, experiment_dir)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            path=experiment_dir / f'{epoch:04}.pth',
        )
        prune_old_checkpoints(eval_hist_, experiment_dir)

        if end_epoch_callback is not None:
            end_epoch_callback()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    path: Path,
) -> None:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'random_state': delu.random.get_state(),
    }
    torch.save(checkpoint, path)


def maybe_load_last_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    experiment_dir: Path,
) -> CheckpointFeatures | None:
    checkpoint_path = last_checkpoint_path(experiment_dir)
    if checkpoint_path is None:
        return None

    checkpoint_features = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=checkpoint_path,
    )
    assert checkpoint_features.epoch == int(checkpoint_path.stem)
    return checkpoint_features


def maybe_load_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    choose_by_metric: bool,
    metric_higher_is_better: bool,
    experiment_dir: Path,
) -> CheckpointFeatures | None:
    eval_hist = maybe_load_eval_hist(experiment_dir)
    if eval_hist is None:
        return None

    if choose_by_metric:
        if metric_higher_is_better:
            best_epoch = int(eval_hist.valid_metric.argmax())
        else:
            best_epoch = int(eval_hist.valid_metric.argmin())
    else:
        best_epoch = int(eval_hist.valid_loss.argmin())

    best_checkpoint_path = experiment_dir / f'{best_epoch:04}.pth'

    checkpoint_features = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=best_checkpoint_path,
    )
    assert checkpoint_features.epoch == best_epoch
    return checkpoint_features


def last_checkpoint_path(experiment_dir: Path) -> Path | None:
    checkpoint_paths = sorted(
        f for f in experiment_dir.glob('*.pth') if f.stem.isnumeric()
    )
    if checkpoint_paths:
        return checkpoint_paths[-1]
    else:
        return None


def last_epoch(experiment_dir: Path) -> int | None:
    checkpoint_path = last_checkpoint_path(experiment_dir)
    if checkpoint_path is None:
        return None
    else:
        return int(checkpoint_path.stem)


def load_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    path: Path,
) -> CheckpointFeatures:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        if epoch + 1 != scheduler.last_epoch:
            warning(
                f'Chekpoint {epoch=} does not match {scheduler.last_epoch=}'
            )

    delu.random.set_state(checkpoint['random_state'])
    return CheckpointFeatures(epoch=epoch)


def maybe_load_eval_hist(experiment_dir: Path) -> eval_hist.EvalHist | None:
    path = experiment_dir / 'eval_hist.pth'
    if not path.exists():
        return None

    return eval_hist.load_eval_hist(path)


def save_eval_hist(hist: eval_hist.EvalHist, experiment_dir: Path) -> None:
    eval_hist.save_eval_hist(hist, experiment_dir / 'eval_hist.pth')


def prune_old_checkpoints(
    hist: eval_hist.EvalHist, experiment_dir: Path
) -> None:
    ix_best_loss = hist.valid_loss.argmin()
    ix_best_metric = hist.valid_metric.argmax()
    ix_last = len(hist.valid_loss) - 1
    ix_keep = list(
        set([ix_last, ix_last - 1, int(ix_best_loss), int(ix_best_metric)])
    )
    checkpoint_paths = [
        f for f in experiment_dir.glob('*.pth') if f.stem.isnumeric()
    ]
    for f in checkpoint_paths:
        epoch = int(f.stem)
        if epoch not in ix_keep:
            f.unlink()


__all__ = [
    'train',
    'load_checkpoint',
    'last_epoch',
    'maybe_load_eval_hist',
    'maybe_load_best_checkpoint',
    'Dataset',
    'DatasetStats',
    'DEVICE',
]
