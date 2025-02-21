import ast
import collections
import contextlib
import inspect
import logging
import os
import re
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from file_io import PathManager
from omegaconf import DictConfig, OmegaConf, open_dict
from data_utils import numpy_seed

logger = logging.getLogger(__name__)


def save_checkpoint(cfg, trainer, epoch_itr, val_loss):
    os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return None

    if not cfg.no_save_optimizer_state:
        trainer.consolidate_optimizer()

    extra_state = {
        "train_iterator": epoch_itr.state_dict(),
        "val_loss": val_loss,
    }

    if hasattr(trainer.task, "get_checkpoint_dict"):
        extra_state.update(trainer.task.get_checkpoint_dict())
        logger.info(f"State of {trainer.task.__class__.__name__} is ready to be persisted with the checkpoint")

    if hasattr(save_checkpoint, "best"):
        extra_state["best"] = save_checkpoint.best

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds[f"checkpoint{epoch_itr.epoch}.pt"] = (
        epoch_itr.end_of_epoch() and not cfg.no_epoch_checkpoints and epoch_itr.epoch % cfg.save_interval == 0
    )
    checkpoint_conds[f"checkpoint_{epoch_itr.epoch}_{trainer.get_num_updates()}.pt"] = (
        not epoch_itr.end_of_epoch()
        and cfg.save_interval_updates > 0
        and trainer.get_num_updates() % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best.pt"] = val_loss is not None and (
        not hasattr(save_checkpoint, "best") or save_checkpoint.best == val_loss
    )
    checkpoint_conds["checkpoint_last.pt"] = not cfg.no_last_checkpoints

    checkpoints = [os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    saved_cp = None
    
    if checkpoints:
        saved_cp = trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            if cfg.write_checkpoints_asynchronously:
                logger.warning(f"Async write while copying {checkpoints[0]} to {cp}")
            else:
                PathManager.copy(checkpoints[0], cp, overwrite=True)
        logger.info(f"Saved checkpoint {checkpoints[0]} (epoch {epoch_itr.epoch} @ {trainer.get_num_updates()} updates)")

    if not epoch_itr.end_of_epoch() and cfg.keep_interval_updates > 0:
        checkpoints = checkpoint_paths(cfg.save_dir, pattern=r"checkpoint_\d+_(\d+)\.pt")
        for old_chk in checkpoints[cfg.keep_interval_updates:]:
            if PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_last_epochs > 0:
        checkpoints = checkpoint_paths(cfg.save_dir, pattern=r"checkpoint(\d+)\.pt")
        for old_chk in checkpoints[cfg.keep_last_epochs:]:
            if PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_best_checkpoints > 0:
        checkpoints = checkpoint_paths(cfg.save_dir, pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(cfg.best_checkpoint_metric))
        checkpoints = checkpoints[::-1] if not cfg.maximize_best_checkpoint_metric else checkpoints
        for old_chk in checkpoints[cfg.keep_best_checkpoints:]:
            if PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    return saved_cp


def load_checkpoint(cfg, trainer, **passthrough_args):
    reset_optimizer = cfg.reset_optimizer
    reset_lr_scheduler = cfg.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(cfg.optimizer_overrides)
    reset_meters = cfg.reset_meters
    reset_dataloader = cfg.reset_dataloader

    checkpoint_path = cfg.restore_file
    if cfg.restore_file == "checkpoint_last.pt":
        checkpoint_path = os.path.join(cfg.save_dir, "checkpoint_last.pt")
        if not PathManager.exists(checkpoint_path) and cfg.finetune_from_model:
            checkpoint_path = cfg.finetune_from_model
            reset_optimizer = reset_lr_scheduler = reset_meters = reset_dataloader = True

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    if extra_state and "best" in extra_state:
        save_checkpoint.best = extra_state["best"]

    epoch_itr = trainer.get_train_iterator(epoch=1, load_dataset=True, **passthrough_args)
    if extra_state and not reset_dataloader:
        epoch_itr.load_state_dict(extra_state["train_iterator"])
        if hasattr(trainer.task, "set_checkpoint_dict"):
            trainer.task.set_checkpoint_dict(extra_state.get(trainer.task.__class__.__name__, {}))

    trainer.lr_step(epoch_itr.epoch)
    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path):
    local_path = PathManager.get_local_path(path)
    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    if "cfg" in state and state["cfg"]:
        state["cfg"] = OmegaConf.create(state["cfg"], flags={"allow_objects": True})

    return _upgrade_state_dict(state)


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt"):
    pt_regexp = re.compile(pattern)
    files = PathManager.ls(path)
    entries = [(int(m.group(1)), f) for f in files if (m := pt_regexp.fullmatch(f))]
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(obj, filename):
    if PathManager.supports_rename(filename):
        with PathManager.open(filename + ".tmp", "wb") as f:
            torch.save(obj, f)
        PathManager.rename(filename + ".tmp", filename)
    else:
        with PathManager.open(filename, "wb") as f:
            torch.save(obj, f)


def verify_checkpoint_directory(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    temp_file = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file, "w"):
            pass
        os.remove(temp_file)
    except OSError as e:
        logger.error(f"Checkpoint directory {save_dir} inaccessible: {e}")
        raise


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"].get("epoch", 0),
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }

    if "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
        with open_dict(cfg):
            # any upgrades for Hydra-based configs
            if (
                "task" in cfg
                and "eval_wer_config" in cfg.task
                and isinstance(cfg.task.eval_wer_config.print_alignment, bool)
            ):
                cfg.task.eval_wer_config.print_alignment = "hard"
            if "generation" in cfg and isinstance(cfg.generation.print_alignment, bool):
                cfg.generation.print_alignment = (
                    "hard" if cfg.generation.print_alignment else None
                )
            if (
                "model" in cfg
                and "w2v_args" in cfg.model
                and cfg.model.w2v_args is not None
                and (
                    hasattr(cfg.model.w2v_args, "task") or "task" in cfg.model.w2v_args
                )
                and hasattr(cfg.model.w2v_args.task, "eval_wer_config")
                and cfg.model.w2v_args.task.eval_wer_config is not None
                and isinstance(
                    cfg.model.w2v_args.task.eval_wer_config.print_alignment, bool
                )
            ):
                cfg.model.w2v_args.task.eval_wer_config.print_alignment = "hard"

    return state


def prune_state_dict(state_dict, model_cfg: Optional[DictConfig]):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            int(layer_string) for layer_string in layers_to_keep.split(",")
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if isinstance(model_cfg, DictConfig):
        context = open_dict(model_cfg)
    else:
        context = contextlib.ExitStack()
    with context:
        if hasattr(model_cfg, "encoder_layers_to_keep"):
            model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"):
            model_cfg.decoder_layers_to_keep = None

    return new_state_dict


def save_ema_as_checkpoint(src_path, dst_path):
    state = load_ema_from_checkpoint(src_path)
    torch_persistent_save(state, dst_path)


def load_ema_from_checkpoint(fpath):
    """Loads exponential moving averaged (EMA) checkpoint from input and
    returns a model with ema weights.

    Args:
      fpath: A string path of checkpoint to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    new_state = None

    with PathManager.open(fpath, "rb") as f:
        new_state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

        # EMA model is stored in a separate "extra state"
        model_params = new_state["extra_state"]["ema"]

        for key in list(model_params.keys()):
            p = model_params[key]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if key not in params_dict:
                params_dict[key] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                raise ValueError("Key {} is repeated in EMA model params.".format(key))

        if len(params_dict) == 0:
            raise ValueError(
                f"Input checkpoint path '{fpath}' does not contain "
                "ema model weights, is this model trained with EMA?"
            )

    new_state["model"] = params_dict
    return new_state
