# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from omegaconf import MISSING
from torch.nn.parallel import DistributedDataParallel

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    CheckpointRetentionPolicy,
    cleanup_checkpoint,
    find_latest_checkpoint,
    keep_last_n_checkpoints,
)
from dinov3.data import SamplerType, make_data_loader, make_dataset
from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
    make_classification_train_transform,
)
from dinov3.eval.data import get_num_classes
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.metrics import build_classification_metric
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.eval.utils import ModelWithIntermediateLayers, evaluate
from dinov3.eval.utils import save_results as default_save_results_func
from dinov3.logging import MetricLogger, SmoothedValue
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-linear.csv"
MAIN_METRICS = [".*_accuracy(_mean)?"]
DEFAULT_SEED = 0


_DEFAULT_LR_LIST: Tuple[float, ...] = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 7e-2, 0.1, 0.15, 0.2)


@dataclass
class TrainConfig:
    dataset: str = MISSING  # train dataset path
    val_dataset: str = MISSING  # val dataset path
    batch_size: int = 128  # batch size (per GPU)
    num_workers: int = 8
    # Linear Head Parameters
    learning_rates: Tuple[float, ...] = _DEFAULT_LR_LIST  # learning rates to grid search
    n_last_blocks_list: Tuple[int] = (1,)  # number of backbone last blocks used for the linear classifier
    # Pooling variants to sweep. True concatenates the CLS token with the mean of the
    # patch tokens; False probes the CLS token alone.
    use_avgpool_list: Tuple[bool, ...] = (True,)
    epochs: int = 10  # number of training epochs
    epoch_length: int = 10010  # length of an epoch in number of iterations
    save_checkpoint_iterations: int | None = (
        None  # number of iterations between two checkpoint saves (default: one epoch)
    )
    eval_period_iterations: int | None = None  # number of iterations between two evaluations (default: one epoch)
    checkpoint_retention_policy: CheckpointRetentionPolicy = CheckpointRetentionPolicy.NONE  # keep checkpoints or not
    resume: bool = True  # whether to resume from existing checkpoints
    classifier_fpath: Optional[str] = None  # path to a file containing pretrained linear classifiers


@dataclass
class EvalConfig:
    batch_size: int = 256  # batch size (per GPU)
    num_workers: int = 8


@dataclass
class TransformConfig:
    resize_size: int = RESIZE_DEFAULT_SIZE
    crop_size: int = CROP_DEFAULT_SIZE


@dataclass
class LinearEvalConfig:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    save_results: bool = False  # save predictions and targets in the output directory
    seed: int = DEFAULT_SEED  # seeds classifier init, training data order, augmentations
    output_dir: str = ""


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_world_size()) / 256.0


def setup_linear_classifiers(
    sample_output,
    n_last_blocks_list,
    learning_rates,
    batch_size,
    num_classes=1000,
    use_avgpool_list=(True,),
):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in use_avgpool_list:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                # 6 decimals, not 5: at small `batch_size * world_size` neighbouring
                # entries of the LR grid can round to the same 5-decimal key, which
                # silently drops one of them from the ModuleDict (it keeps its optimizer
                # param group but never receives gradients).
                key = f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.6f}".replace(".", "_")
                assert key not in linear_classifiers_dict, (
                    f"Learning rates {_lr} and another entry collide on key '{key}'; "
                    "widen the format or drop the duplicate."
                )
                linear_classifiers_dict[key] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


def make_eval_transform(config: TransformConfig):
    if config.resize_size / config.crop_size != 256 / 224:
        logger.warning(
            f"Default resize / crop ratio is 256 / 224, here we have {config.resize_size} / {config.crop_size}"
        )
    transform = make_classification_eval_transform(resize_size=config.resize_size, crop_size=config.crop_size)
    return transform


def make_eval_data_loader(
    *,
    test_dataset_str,
    transform_config,
    batch_size,
    num_workers,
):
    transform = make_eval_transform(transform_config)
    test_dataset = make_dataset(dataset_str=test_dataset_str, transform=transform)

    class_mapping = None
    if hasattr(test_dataset, "get_imagenet_class_mapping"):
        class_mapping = test_dataset.get_imagenet_class_mapping()

    test_data_loader = make_data_loader(
        dataset=DatasetWithEnumeratedTargets(test_dataset, pad_dataset=True, num_replicas=distributed.get_world_size()),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )
    return test_data_loader, class_mapping


@dataclass
class Evaluator:
    batch_size: int
    num_workers: int
    transform_config: TransformConfig
    dataset_str: str
    metrics_file_path: str
    training_num_classes: int
    save_results_func: Optional[Callable]

    def __post_init__(self):
        self.data_loader, self.class_mapping = make_eval_data_loader(
            test_dataset_str=self.dataset_str,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transform_config=self.transform_config,
        )
        self.main_metric_name = f"{self.dataset_str}_accuracy"

    @torch.no_grad()
    def _evaluate_linear_classifiers(
        self,
        *,
        feature_model,
        linear_classifiers,
        iteration,
        prefixstring="",
        best_classifier_on_val=None,
        accumulate_results=False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        logger.info("running validation !")

        num_classes = len(self.class_mapping) if self.class_mapping is not None else self.training_num_classes
        metric = build_classification_metric(num_classes=num_classes)
        postprocessors = {
            k: LinearPostprocessor(v, self.class_mapping) for k, v in linear_classifiers.classifiers_dict.items()
        }
        metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

        _, results_dict_temp, accumulated_results = evaluate(
            feature_model,
            self.data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
            accumulate_results=accumulate_results,
        )

        logger.info("")
        results_dict = {}
        max_accuracy = 0
        best_classifier = ""
        for _, (classifier_string, metric) in enumerate(results_dict_temp.items()):
            logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
            if (
                best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
            ) or classifier_string == best_classifier_on_val:
                max_accuracy = metric["top-1"].item()
                best_classifier = classifier_string

        results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}

        logger.info(f"best classifier: {results_dict['best_classifier']}")

        accumulated_best_results = None
        if accumulated_results is not None:
            accumulated_best_results = accumulated_results[best_classifier]

        if distributed.is_main_process():
            with open(self.metrics_file_path, "a") as f:
                f.write(f"iter: {iteration}\n")
                for k, v in results_dict.items():
                    f.write(json.dumps({k: v}) + "\n")
                f.write("\n")

        return results_dict, accumulated_best_results

    def evaluate_and_maybe_save(
        self,
        feature_model,
        linear_classifiers,
        iteration: int,
        best_classifier_on_val: Optional[Any] = None,
        save_filename_suffix: str = "",
        prefixstring: str = "",
    ):
        logger.info(f"Testing on {self.dataset_str}")
        save_results = self.save_results_func is not None
        full_results_dict, accumulated_best_results = self._evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=remove_ddp_wrapper(linear_classifiers),
            iteration=iteration,
            prefixstring=prefixstring,
            best_classifier_on_val=best_classifier_on_val,
            accumulate_results=save_results,
        )
        if self.save_results_func is not None:
            self.save_results_func(
                filename_suffix=f"{self.dataset_str}{save_filename_suffix}", **accumulated_best_results
            )

        results_dict = {
            self.main_metric_name: 100.0 * full_results_dict["best_classifier"]["accuracy"],
            "best_classifier": full_results_dict["best_classifier"]["name"],
        }
        return results_dict


def setup_linear_training(
    *,
    config: TrainConfig,
    sample_output: torch.Tensor,
    training_num_classes: int,
    checkpoint_output_dir: str,
):
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        config.n_last_blocks_list,
        config.learning_rates,
        config.batch_size,
        training_num_classes,
        use_avgpool_list=config.use_avgpool_list,
    )
    max_iter = config.epochs * config.epoch_length
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    start_iter = 0
    best_accuracy = -1
    if config.resume and (
        last_checkpoint_dir := find_latest_checkpoint(config.classifier_fpath or checkpoint_output_dir)
    ):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        checkpoint = torch.load(last_checkpoint_dir / "checkpoint.pth")
        start_iter = checkpoint.get("iteration", -1) + 1
        best_accuracy = checkpoint.get("best_accuracy", -1)
        linear_classifiers.load_state_dict(checkpoint["linear_classifiers"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Legacy checkpoints were saved after both optimizer and scheduler steps.
            scheduler.load_state_dict({
                "last_epoch": start_iter,
                "_step_count": start_iter + 1,
                "base_lrs": [group["initial_lr"] for group in optimizer.param_groups],
                "_last_lr": [group["lr"] for group in optimizer.param_groups],
            })

    criterion = nn.CrossEntropyLoss()

    return (
        linear_classifiers,
        start_iter,
        max_iter,
        criterion,
        optimizer,
        scheduler,
        best_accuracy,
    )


def train_linear_classifiers(
    *,
    feature_model,
    train_dataset,
    train_config: TrainConfig,
    training_num_classes: int,
    val_evaluator: Evaluator,
    checkpoint_output_dir: str,
    seed: int = DEFAULT_SEED,
):
    (linear_classifiers, start_iter, max_iter, criterion, optimizer, scheduler, best_accuracy,) = setup_linear_training(
        config=train_config,
        sample_output=feature_model(train_dataset[0][0].unsqueeze(0).cuda()),
        training_num_classes=training_num_classes,
        checkpoint_output_dir=checkpoint_output_dir,
    )
    checkpoint_period = train_config.save_checkpoint_iterations or train_config.epoch_length
    eval_period = train_config.eval_period_iterations or train_config.epoch_length
    ckpt_dir = Path(checkpoint_output_dir).expanduser()

    sampler_type = SamplerType.INFINITE
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter * train_config.batch_size,
        drop_last=True,
        persistent_workers=train_config.num_workers > 0,
    )

    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6g}"))
    header = "Training"
    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)

        if len(labels.shape) > 1:
            labels = labels.float()
        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Checkpointing
        is_last_iteration = (iteration + 1) == max_iter
        is_ckpt_iteration = ((iteration + 1) % checkpoint_period == 0) or is_last_iteration
        if is_ckpt_iteration:
            if distributed.is_subgroup_main_process():
                ckpt_sub_dir = "final" if is_last_iteration else str(iteration)
                (ckpt_dir / ckpt_sub_dir).mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "iteration": iteration,
                    "linear_classifiers": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                }
                torch.save(checkpoint, ckpt_dir / ckpt_sub_dir / "checkpoint.pth")
                keep_last_n_checkpoints(ckpt_dir, train_config.checkpoint_retention_policy.max_to_keep)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            val_results_dict = val_evaluator.evaluate_and_maybe_save(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                prefixstring=f"ITER: {iteration}",
                iteration=iteration,
            )
            val_accuracy = val_results_dict[val_evaluator.main_metric_name]
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                (ckpt_dir / "best").mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "iteration": iteration,
                    "linear_classifiers": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                }
                torch.save(checkpoint, ckpt_dir / "best" / "checkpoint.pth")
            torch.distributed.barrier()

        iteration = iteration + 1

    return feature_model, linear_classifiers, iteration


def make_train_transform(config: TransformConfig):
    train_transform = make_classification_train_transform(crop_size=config.crop_size)
    return train_transform


def make_train_dataset(train_dataset: str, transform_config: TransformConfig):
    train_transform = make_train_transform(transform_config)
    return make_dataset(dataset_str=train_dataset, transform=train_transform)


def eval_linear_with_model(*, model: torch.nn.Module, autocast_dtype, config: LinearEvalConfig):
    start = time.time()
    cudnn.benchmark = True
    train_dataset = make_train_dataset(config.train.dataset, config.transform)
    training_num_classes = get_num_classes(train_dataset)
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, max(config.train.n_last_blocks_list), autocast_ctx)
    save_results_func = partial(default_save_results_func, output_dir=config.output_dir) if config.save_results else None
    evaluator = Evaluator(
        dataset_str=config.train.val_dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
        transform_config=config.transform,
        metrics_file_path=os.path.join(config.output_dir, "results_eval_linear.json"),
        training_num_classes=training_num_classes,
        save_results_func=save_results_func,
    )
    checkpoint_output_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    feature_model, linear_classifiers, iteration = train_linear_classifiers(
        feature_model=feature_model,
        train_dataset=train_dataset,
        train_config=config.train,
        training_num_classes=training_num_classes,
        val_evaluator=evaluator,
        checkpoint_output_dir=checkpoint_output_dir,
        seed=config.seed,
    )
    results = evaluator.evaluate_and_maybe_save(
        feature_model=feature_model, linear_classifiers=linear_classifiers, iteration=iteration,
    )
    if distributed.is_subgroup_main_process():
        cleanup_checkpoint(checkpoint_output_dir, config.train.checkpoint_retention_policy)
    logger.info(f"Linear evaluation done in {int(time.time() - start)}s")
    return results


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=LinearEvalConfig)
    model, model_context = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    results_dict = eval_linear_with_model(
        model=model, config=dataclass_config, autocast_dtype=model_context["autocast_dtype"]
    )
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    # Seeds the process RNGs (classifier init, dataloader worker seeds) before anything
    # random happens; `benchmark_launcher` re-reads the same value off the parsed config
    # for the training sampler.
    with job_context(output_dir=eval_args["output_dir"], seed=int(eval_args.get("seed", DEFAULT_SEED))):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    main()
