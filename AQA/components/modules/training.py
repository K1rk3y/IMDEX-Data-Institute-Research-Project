# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import logging
import os
import sys
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from fairseq import checkpoint_utils, models, optim, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.models.ema import build_ema
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler
from fairseq.utils import safe_hasattr

logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for single GPU training."""

    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        if isinstance(cfg, Namespace):
            logger.warning(
                "argparse.Namespace configuration is deprecated! Automatically converting to OmegaConf"
            )
            cfg = convert_namespace_to_omegaconf(cfg)

        self.cfg = cfg
        self.task = task

        # Catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Move model and criterion to device
        self._criterion = criterion.to(device=self.device)
        self._model = model.to(device=self.device)

        # Check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = None
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None
        self._ema = None

        self.quantizer = quantizer
        if self.quantizer is not None:
            self.quantizer.set_trainer(self)

        # Log CUDA environment for single GPU
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            self.cuda_env_arr = [self.cuda_env]
            utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)
        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Always save checkpoints on single GPU"""
        return True

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def criterion(self):
        return self._criterion

    @property
    def model(self):
        return self._model

    @property
    def ema(self):
        if self._ema is None:
            self._build_ema()
        return self._ema

    def _build_ema(self):
        if self.cfg.ema.store_ema:
            self._ema = build_ema(self._model, self.cfg.ema, self.device)
            logger.info("Exponential Moving Average Shadow Model is initialized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        if (
            self.cfg.optimization.debug_param_names
            and self.cfg.common.fp16_no_flatten_grads
        ):
            params = []
            self.param_names = []
            for n, p in chain(
                self.model.named_parameters(), self.criterion.named_parameters()
            ):
                if p.requires_grad:
                    params.append(p)
                    self.param_names.append(n)
        else:
            params = list(
                filter(
                    lambda p: p.requires_grad,
                    chain(self.model.parameters(), self.criterion.parameters()),
                )
            )

        # Mixed precision handling
        if self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: Your device may not support faster training with --fp16/--bf16/--amp. "
                    "Consider using FP32 for better performance."
                )
            
            if self.cfg.common.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.cfg, params
                )
            elif self.cfg.common.amp:
                self._optimizer = optim.AMPOptimizer.build_optimizer(self.cfg, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            self._optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        # Initialize learning rate scheduler
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            self.optimizer,
        )
        self._lr_scheduler.step_update(0)

    def state_dict(self):
        state_dict = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True),
            "model": self.model.state_dict(),
            "criterion": (
                self.criterion.state_dict()
                if utils.has_parameters(self.criterion)
                else None
            ),
            "optimizer_history": (self._optim_history or []) + [{
                "criterion_name": self.get_criterion().__class__.__name__,
                "optimizer_name": self.optimizer.__class__.__name__,
                "lr_scheduler_state": self.lr_scheduler.state_dict(),
                "num_updates": self.get_num_updates(),
            }],
            "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
        }
        
        if self.cfg.ema.store_ema:
            state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
            if self.cfg.ema.ema_fp32:
                state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params
        
        if not self.cfg.checkpoint.no_save_optimizer_state:
            state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        
        return state_dict

    def save_checkpoint(self, filename, extra_state):
        """Save training state to checkpoint file"""
        logger.info(f"Saving checkpoint to {filename}")
        state_dict = utils.move_to_cpu(self.state_dict())
        state_dict["extra_state"].update(extra_state)
        
        checkpoint_utils.torch_persistent_save(
            state_dict,
            filename,
            async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
        )
        logger.info(f"Checkpoint saved to {filename}")
        return filename

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load training state from checkpoint file"""
        extra_state, self._optim_history, last_optim_state = None, [], None
        logger.info(f"Loading checkpoint from {filename}")

        if PathManager.isfile(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)
            last_optim_state = state.get("last_optimizer_state", None)

            # Load model parameters
            try:
                self.model.load_state_dict(
                    state["model"], strict=True, model_cfg=self.cfg.model
                )
                del state["model"]
                
                if utils.has_parameters(self.criterion):
                    self.criterion.load_state_dict(state["criterion"], strict=True)
                    del state["criterion"]
                    
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model parameters from {filename}: {str(e)}\n"
                    "Please ensure the model architecture matches the checkpoint."
                ) from e

            # Load training state
            extra_state = state.get("extra_state", {})
            self._optim_history = state.get("optimizer_history", [])
            
            # Handle optimizer state
            if last_optim_state and not reset_optimizer:
                self._build_optimizer()
                last_optim = self._optim_history[-1] if self._optim_history else {}
                
                # Validate compatibility
                if last_optim.get("criterion_name") != self.criterion.__class__.__name__:
                    logger.warning("Criterion mismatch - optimizer state may be invalid")
                
                if last_optim.get("optimizer_name") != self.optimizer.__class__.__name__:
                    logger.warning("Optimizer mismatch - state may not load properly")

                if not reset_lr_scheduler:
                    self.lr_scheduler.load_state_dict(
                        last_optim.get("lr_scheduler_state", {})
                    )

                self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)
                self.set_num_updates(last_optim.get("num_updates", 0))

            # Restore training metrics and timers
            if extra_state:
                self._restore_training_state(extra_state, reset_meters)

        else:
            logger.warning(f"Checkpoint not found at {filename}")
            return None

        return extra_state

    def _restore_training_state(self, extra_state, reset_meters):
        """Restore miscellaneous training state elements"""
        # Timing information
        self._previous_training_time = extra_state.get(
            "previous_training_time", self._previous_training_time
        )
        self._start_time = time.time()

        # Metrics restoration
        if not reset_meters and "metrics" in extra_state:
            metrics.load_state_dict(extra_state["metrics"])
            # Reset timing metrics for clean measurement
            for meter in metrics.get_meters("default"):
                if isinstance(meter, meters.TimeMeter):
                    meter.reset()

        # EMA handling
        if self.cfg.ema.store_ema:
            if "ema" in extra_state:
                self.ema.restore(extra_state["ema"], build_fp32_params=False)
                if self.cfg.ema.ema_fp32:
                    self.ema.build_fp32_params(
                        extra_state.get("ema_fp32_params", None)
                    )
            else:
                logger.warning(
                    "EMA configuration active but no EMA state found in checkpoint. "
                    "Initializing new EMA parameters."
                )

        logger.info(
            f"Resumed training from checkpoint (epoch {extra_state.get('epoch', 'unknown')} "
            f"| updates {self.get_num_updates()})"
        )

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=False,  # Default to False since we're not using shards
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.cfg.dataset.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
                tpu=False,  # Explicitly set to False as we're not using TPU
            )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.train_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.cfg.dataset.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=(self.cfg.common.seed + epoch)
            if self.cfg.dataset.update_ordered_indices_seed
            else self.cfg.common.seed,
            num_shards=1,  # Single GPU, so only one shard
            shard_id=0,   # Only one shard, so ID is 0
            num_workers=self.cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=self.cfg.optimization.skip_remainder_batch,
            grouped_shuffling=self.cfg.dataset.grouped_shuffling,
            update_epoch_batch_itr=self.cfg.dataset.update_epoch_batch_itr,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=1,  # Single GPU, so only one shard
            shard_id=0,   # Only one shard, so ID is 0
            num_workers=self.cfg.dataset.num_workers,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        if self.quantizer is not None:
            self.quantizer.begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        self.task.begin_valid_epoch(epoch, self.get_model())

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        has_oom = False

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):  # delayed update loop
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                        **extra_kwargs,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    has_oom = True
                    if raise_oom:
                        raise e
                else:
                    raise e
            except Exception:
                self.consolidate_optimizer()
                self.save_checkpoint(
                    os.path.join(self.cfg.checkpoint.save_dir, "crash.pt"), {}
                )
                raise

            if has_oom:
                logger.warning(
                    "attempting to recover from OOM in forward/backward pass"
                )
                ooms += 1
                self.zero_grad()
                if self.cuda:
                    torch.cuda.empty_cache()

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # No need for gradient reduction on a single GPU
        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                pass  # Removed all_reduce_grads calls

            with torch.autograd.profiler.record_function("multiply-grads"):
                self.optimizer.multiply_grads(1.0 / (sample_size or 1.0))

            with torch.autograd.profiler.record_function("clip-grads"):
                grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)

            if not torch.isfinite(grad_norm).all():
                raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                self.task.optimizer_step(
                    self.optimizer, model=self.model, update_num=self.get_num_updates()
                )

        except FloatingPointError:
            self.consolidate_optimizer()
            self.save_checkpoint(
                os.path.join(self.cfg.checkpoint.save_dir, "crash.pt"), {}
            )
            raise

        except OverflowError as e:
            overflow = True
            logger.info(f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}")
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()

        if self.cuda and self.cuda_env is not None:
            gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
            gb_free = self.cuda_env.total_memory_in_GB - gb_used
            metrics.log_scalar("gb_free", gb_free, priority=1500, round=1, weight=0)

        logging_output = self._reduce_and_log_stats(
            logging_outputs, sample_size, grad_norm
        )

        if self.cfg.common.fp16 or self.cfg.common.amp:
            metrics.log_scalar(
                "loss_scale",
                (
                    self.optimizer.scaler.loss_scale
                    if self.cfg.common.fp16
                    else self.optimizer.scaler.get_scale()
                ),
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """[deprecated] Get a specific meter by name."""
        from fairseq import meters

        if "get_meter" not in self._warn_once:
            self._warn_once.add("get_meter")
            utils.deprecation_warning(
                "Trainer.get_meter is deprecated. Please use fairseq.metrics instead."
            )

        train_meters = metrics.get_meters("train")
        if train_meters is None:
            train_meters = {}

        if name == "train_loss" and "loss" in train_meters:
            return train_meters["loss"]
        elif name == "train_nll_loss":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = train_meters.get("nll_loss", None)
            return m or meters.AverageMeter()
        elif name == "wall":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = metrics.get_meter("default", "wall")
            return m or meters.TimeMeter()
        elif name == "wps":
            m = metrics.get_meter("train", "wps")
            return m or meters.TimeMeter()
        elif name in {"valid_loss", "valid_nll_loss"}:
            # support for legacy train.py, which assumed these meters
            # are always initialized
            k = name[len("valid_") :]
            m = metrics.get_meter("valid", k)
            return m or meters.AverageMeter()
        elif name == "oom":
            return meters.AverageMeter()
        elif name in train_meters:
            return train_meters[name]
        return None

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        if self.quantizer:
            self.quantizer.step_update(self._num_updates)
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):
        def agg_norm_fn(total_norm):
            total_norm = total_norm.cuda().float() ** 2
            total_norm = distributed_utils.all_reduce(
                total_norm, group=self.data_parallel_process_group
            )
            return total_norm**0.5

        should_agg_norm = self.is_fsdp and (
            self.data_parallel_process_group is not None
            or torch.distributed.is_initialized()
        )
        return self.optimizer.clip_grad_norm(
            clip_norm, aggregate_norm_fn=agg_norm_fn if should_agg_norm else None
        )

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        if self.cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        return sample

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        # Given that PCIe/NVLink bandwidth is significantly smaller than DRAM bandwidth
        # it makes sense to do the format conversion on the CPU and then transfer
        # a smaller buffer to the device. This also saves GPU memory capacity.

        if self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self.cuda:
            if self.pipeline_model_parallel:
                if "target" in sample:
                    sample["target"] = utils.move_to_cuda(
                        sample["target"], device=self.last_device
                    )
            else:
                sample = utils.move_to_cuda(sample)
        elif self.tpu and is_dummy:
            # the dummy batch may not be on the appropriate device
            sample = utils.move_to_cuda(sample, device=self.device)

        if not self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        if self.data_parallel_world_size == 1:
            return False
        elif self.cfg.optimization.use_bmuf:
            return (
                self.get_num_updates() + 1
            ) % self.cfg.bmuf.global_sync_iter == 0 and (
                self.get_num_updates() + 1
            ) > self.cfg.bmuf.warmup_iterations
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if self.tpu:
            raise NotImplementedError
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data, device=self.device, group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf, group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    (
                        torch.isfinite(tensor).all()
                        and (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()
                    )
                    or (self.cfg.common.amp and not torch.isfinite(tensor).all())
                    # in case of amp non-finite grads are fine
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(
                    pretty_detail
                )
                # use FloatingPointError to trigger NanDetector
                raise FloatingPointError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=legacy_ddp. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.cfg.optimization.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs

            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            if self.tpu:
                logging_output = {}
            else:
                logging_output = agg.get_smoothed_values()
                logging_output["sample_size"] = sample_size
                for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                    if key_to_delete in logging_output:
                        del logging_output[key_to_delete]
            return logging_output

    def _check_xla_compilation(self):
        import torch_xla.debug.metrics as met

        compile_stats = met.metric_data("CompileTime")
        if compile_stats is None:
            return
        num_xla_compiles = compile_stats[0]
        if num_xla_compiles > self._num_xla_compiles:
            logger.warning(
                "XLA compilation detected on device #{}; too many of these can lead "
                "to slow training, but we expect a few in the beginning".format(
                    self.cfg.distributed_training.distributed_rank
                )
            )
        self._num_xla_compiles = num_xla_compiles

    def _xla_markstep_and_send_to_cpu(self, data=None):
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        if data is not None:
            from fairseq.utils import xla_device_to_cpu

            return xla_device_to_cpu(data)


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)