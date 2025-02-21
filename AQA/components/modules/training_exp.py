import logging
import os
import sys
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from optim_factory import LayerDecayValueAssigner, create_optimizer
from iterators import EpochBatchIterator

import checkpoint_utils
from file_io import PathManager
from components.logging import meters, metrics
from ema import build_ema
from nan_detector import NanDetector


from batching import batch_by_size
import utils

logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for single GPU training."""

    def __init__(self, cfg, task, model, criterion, quantizer=None):
        self.cfg = cfg
        self.task = task

        # Move model and criterion to device
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self._criterion = criterion.to(device=self.device)
        self._model = model.to(device=self.device)

        self._dummy_batch = None
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self.quantizer = quantizer
        if self.quantizer is not None:
            self.quantizer.set_trainer(self)

        # Log CUDA environment for single GPU
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
        else:
            self.cuda_env = None

        metrics.log_start_time("wall", priority=790, round=0)
        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

        self.dataset_to_epoch_iter = dict()
        self.DatasetLoader = 

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        return True

    @property
    def checkpoint_suffix(self) -> str:
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
            self._build_optimizer()
        return self._lr_scheduler

    def _build_optimizer(self):
        # Calculate total layers correctly
        encoder_layers = len(self.model.encoder.layer)
        num_layers = encoder_layers + 2  # +2 for embeddings & final layers
        
        layer_decay_values = [self.cfg.layer_decay ** (num_layers - i) 
                            for i in range(num_layers)]
        assigner = LayerDecayValueAssigner(layer_decay_values)
        get_num_layer = assigner.get_layer_id
        get_layer_scale = assigner.get_scale

        # Create optimizer with corrected layers
        self._optimizer = create_optimizer(
            args=self.cfg,
            model=self.model,
            criterion=self.criterion,
            get_num_layer=get_num_layer,
            get_layer_scale=get_layer_scale,
            filter_bias_and_bn=True,
            skip_list=None
        )

        # Initialize learning rate scheduler (example using timm's scheduler)
        from timm.scheduler import create_scheduler
        self._lr_scheduler, _ = create_scheduler(self.cfg, self.optimizer)
        
        # Initialize scheduler step for iteration-based updates
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
        extra_state, self._optim_history = None, []
        logger.info(f"Loading checkpoint from {filename}")

        if PathManager.isfile(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)
            
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
            if state.get("last_optimizer_state") and not reset_optimizer:
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

                self.optimizer.load_state_dict(state["last_optimizer_state"], optimizer_overrides)
                self.set_num_updates(last_optim.get("num_updates", 0))

        else:
            logger.warning(f"Checkpoint not found at {filename}")
            return None

        if extra_state:
            self._restore_training_state(extra_state, reset_meters)

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

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        This function has been adapted to accept standard dataset loader objects (e.g. a GigaSpeechDataset)
        that follow the standard Python dataset protocol. It still uses Fairseq's EpochBatchIterator to
        yield batches.

        Args:
            dataset: dataset to batch. Expected to implement __len__ and __getitem__.
            max_tokens (int, optional): max number of tokens in each batch.
            max_sentences (int, optional): max number of sentences (samples) in each batch.
            max_positions (int, optional): maximum allowed sample length (e.g. number of tokens).
            ignore_invalid_inputs (bool, optional): if True, examples that are too long (w.r.t. max_positions)
                are silently dropped.
            required_batch_size_multiple (int, optional): require batch size to be a multiple of this number.
            seed (int, optional): seed for random number generator for reproducibility.
            num_shards (int, optional): number of shards for the data iterator.
            shard_id (int, optional): which shard to return.
            num_workers (int, optional): number of subprocesses to use for data loading.
            epoch (int, optional): the current epoch number.
            data_buffer_size (int, optional): number of batches to preload.
            disable_iterator_cache (bool, optional): if True, disable caching of the iterator.
            skip_remainder_batch (bool, optional): if True, discard the last batch if it’s smaller than expected.
            grouped_shuffling (bool, optional): if True, perform grouped shuffling.
            update_epoch_batch_itr (bool, optional): if True, force rebuilding the epoch iterator.

        Returns:
            An instance of iterators.EpochBatchIterator that yields batches produced by the dataset's
            collater (if available) or as raw lists.
        """

        # Determine if we can reuse an existing epoch iterator.
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        logger.info(f"can_reuse_epoch_itr = {can_reuse_epoch_itr}")
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        # Initialize the dataset with the correct starting epoch, if supported.
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        def make_batches(dataset, epoch):
            logger.info(f"creating new batches for epoch {epoch}")

            # Define indices over the entire dataset.
            indices = list(range(len(dataset)))

            # Define a function that returns the token count for a given index.
            # Here we assume that the sample's length can be determined via len(sample).
            def num_tokens_fn(idx):
                sample = dataset[idx]
                try:
                    return len(sample)
                except Exception:
                    return 1  # Fallback if sample is not directly sized.

            # Create mini-batches using the provided batch_by_size function.
            batches = batch_by_size(
                indices,
                num_tokens_fn=num_tokens_fn,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            return batches

        # Retrieve configuration options from self.cfg if available.
        reuse_dataloader = getattr(self.cfg, "reuse_dataloader", True)
        persistent_workers = getattr(self.cfg, "persistent_workers", True)
        rebuild_batches = getattr(self.cfg, "rebuild_batches", False)
        logger.info(f"reuse_dataloader = {reuse_dataloader}")
        logger.info(f"rebuild_batches = {rebuild_batches}")

        if rebuild_batches:
            logger.info("batches will be rebuilt for each epoch")
            batch_sampler = make_batches  # a callable that will rebuild batches each time.
        else:
            batch_sampler = make_batches(dataset, epoch)

        # Use the dataset's collater if available; otherwise, default to identity.
        collate_fn = dataset.collater if hasattr(dataset, "collater") else lambda x: x

        # Create the EpochBatchIterator (Fairseq's standard iterator).
        epoch_iter = EpochBatchIterator(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
            reuse_dataloader=reuse_dataloader,
            persistent_workers=persistent_workers,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def get_train_iterator(
        self,
        epoch,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        logger.info("loading train data for epoch {}".format(epoch))
        self.dataset_train = self.DatasetLoader(
            'train',
            root_dir="/path/to/dataset",
            subset="xs",
            vocab_size=10000,
            max_text_length=256
        )
        batch_iterator = self.get_batch_iterator(
            dataset=self.dataset_train,
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=(self.cfg.common.seed + epoch)
            if self.cfg.dataset.update_ordered_indices_seed
            else self.cfg.common.seed,
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
        disable_iterator_cache=False,
    ):
        self.dataset_valid = self.DatasetLoader(
            'val',
            root_dir="/path/to/dataset",
            subset="xs",
            vocab_size=10000,
            max_text_length=256
        )

        batch_iterator = self.get_batch_iterator(
            dataset=self.dataset_valid,
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_workers=self.cfg.dataset.num_workers,
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
        for i, sample in enumerate(samples):  
            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                # forward and backward
                with NanDetector(self.model):
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

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    has_oom = True
                    if raise_oom:
                        raise e
                else:
                    raise e
            except Exception:
                self.save_checkpoint(
                    os.path.join(self.cfg.checkpoint.save_dir, "crash.pt"), {}
                )
                raise

        if is_dummy_batch:
            sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # Multiply gradients by 1/sample_size
        self.optimizer.multiply_grads(1.0 / (sample_size or 1.0))

        try:
            grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)
        except FloatingPointError:
            self.save_checkpoint(
                os.path.join(self.cfg.checkpoint.save_dir, "crash.pt"), {}
            )
            raise

        # Inside train_step after clipping
        self.optimizer.step()
        self.set_num_updates(self.get_num_updates() + 1)

        if self.cfg.ema.store_ema:
            self.ema.step(self.model, self.get_num_updates())

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
        return self.lr_step_update()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
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
        return self.optimizer.clip_grad_norm(clip_norm)

    def cumulative_training_time(self):
        return time.time() - self._start_time + self._previous_training_time

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

        if self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self.cuda:
            sample = utils.move_to_cuda(sample)
        
        if not self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _set_seed(self):
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None:
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

            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output
