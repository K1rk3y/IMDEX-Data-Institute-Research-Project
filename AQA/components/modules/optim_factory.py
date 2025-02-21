import torch
from torch import optim as optim
from itertools import chain

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
# from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_aqa(var_name, num_max_layer):
    parts = var_name.split('.')
    
    # Initial components (embeddings, projections, conv pos)
    if any(parts[0] in {
        "word_embeddings", "text_token_type", "audio_token_type",
        "position_embeddings", "text_position_embeddings",
        "audio_position_embeddings", "feature_extractor",
        "post_extract_proj", "audio_projection", "text_conv_pos",
        "audio_conv_pos", "mask_emb", "mask_emb_text",
        "quantizer", "input_quantizer", "project_q", "project_inp"
    }):
        return 0
    
    # Main encoder layers (shift by +1 to start at layer 1)
    if parts[0] == "encoder" and len(parts) >= 3 and parts[1] == "layer":
        try:
            layer_id = int(parts[2]) + 1  # Shift encoder layers
            return min(layer_id, num_max_layer - 1)
        except ValueError:
            return num_max_layer - 1
    
    # Final components (projections, pooler)
    if any(parts[0] in {"final_proj_text", "final_proj_audio", "pooler", "target_glu"}):
        return num_max_layer - 1
    
    # LayerNorm/dropout in encoder layers (shift by +1)
    if "LayerNorm" in parts or "dropout" in parts:
        if "encoder.layer" in var_name:
            try:
                layer_pos = parts.index("layer")
                layer_id = int(parts[layer_pos + 1]) + 1  # Shift
                return min(layer_id, num_max_layer - 1)
            except (ValueError, IndexError):
                pass
        return 0
    
    # Additional layer (bypass)
    if parts[0] == "additional_layer":
        return num_max_layer - 1
    
    # Default to last layer
    return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_aqa(var_name, len(self.values))


def get_parameter_groups(
        model, criterion, weight_decay=1e-5, skip_list=(), get_num_layer=None, 
        get_layer_scale=None,
    ):
    parameter_group_names = {}
    parameter_group_vars = {}

    named_parameters = list(chain(model.named_parameters(), criterion.named_parameters()))

    for name, param in named_parameters:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
        args, model, criterion, get_num_layer=None, get_layer_scale=None, 
        filter_bias_and_bn=True, skip_list=None
    ):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()

        if hasattr(criterion, 'no_weight_decay'):
            skip.update(criterion.no_weight_decay())

        parameters = get_parameter_groups(
            model, criterion, weight_decay, skip, get_num_layer, get_layer_scale,
        )
        weight_decay = 0.
    else:
        parameters = list(
                filter(
                    lambda p: p.requires_grad,
                    chain(model.parameters(), criterion.parameters()),
                )
            )

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'novograd':
    #     optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
