import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import gc
from functools import partial
from pathlib import Path
from collections import OrderedDict

from utilities.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from utilities.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from utilities.build import build_dataset
from utilities.engine_for_finetuning import train_one_epoch, train_one_epoch_no_dist, validation_one_epoch, validation_one_epoch_no_dist, final_test, final_test_no_dist, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate, def_collate
import utils
import contextlib
import argparse
import copy

from torch.utils import model_zoo
from utilities.celebdf_dataset import CelebDFDataSet

import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
import math

from mamba_ssm.modules.mamba_simple import MambaSL

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'checkpoints'
_MODELS = {
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_k400_f16_res224.pth")
}


import logging

# Configure logging
logging.basicConfig(
    filename='debug.txt',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_gradient_norm(name):
    def hook(grad):
        logging.info(f"Gradient norm of {name}: {grad.norm().item()}")
    return hook


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, valid_positions=None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, valid_positions, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, valid_positions, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(MambaSL, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SpatialNeighborhoodAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def create_neighborhood_mask(self, H, W, device):
        """
        Creates an attention mask where each patch only attends to its 4 immediate neighbors
        Args:
            H: Height in patches
            W: Width in patches
            device: torch device
        Returns:
            mask: Boolean tensor of shape (H*W, H*W)
        """
        mask = torch.zeros(H * W, H * W, dtype=torch.bool, device=device)
        
        for i in range(H * W):
            row = i // W
            col = i % W
            
            # Up neighbor
            if row > 0:
                mask[i, i - W] = True
            # Down neighbor
            if row < H - 1:
                mask[i, i + W] = True
            # Left neighbor
            if col > 0:
                mask[i, i - 1] = True
            # Right neighbor
            if col < W - 1:
                mask[i, i + 1] = True
                
        return mask

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W) from Conv3d
        Returns:
            spatial_embedding: Tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Create attention mask for 4-neighborhood connectivity
        attn_mask = ~self.create_neighborhood_mask(H, W, x.device)
        
        # Reshape to (B*T, H*W, C) for attention
        x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        
        # Run multi-head attention
        spatial_embedding, _ = self.attention(
            query=x_reshaped,
            key=x_reshaped,
            value=x_reshaped,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Reshape back to original format (B, C, T, H, W)
        spatial_embedding = spatial_embedding.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return spatial_embedding


class PatchEmbedWithSpatialContext(nn.Module):
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        # Initialize the original PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        self.spatial_attention = SpatialNeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.0
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Get original patch embeddings using Conv3d
        x = self.patch_embed(x)  # Shape: (B, C, T, H, W)
        
        # Get spatial context embeddings
        spatial_context = self.spatial_attention(x)
        
        # Add spatial context to original embeddings with learnable scale
        enhanced_embedding = x + self.gamma * spatial_context
        
        return enhanced_embedding
    

class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1,
            num_frames=16,  # Set num_frames to match checkpoint
            fc_drop_rate=0.,
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}  # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
        # Model parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames  # Ensure num_frames matches checkpoint

        """self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )"""

        self.patch_embed = PatchEmbedWithSpatialContext(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches  # H * W
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # Mamba layers
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        # Output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        # Original initialization
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.temporal_pos_embedding, std=.02)  # Initialize temporal_pos_embedding
        # Mamba initialization
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.register_patch_embed_hooks()
        self.register_hooks()


    def register_patch_embed_hooks(self):
        for name, param in self.patch_embed.named_parameters():
            param.register_hook(log_gradient_norm(f"patch_embed.{name}"))

    def register_hooks(self):
        # Register hooks for patch embeddings
        self.register_patch_embed_hooks()
        # Register hooks for positional embeddings
        self.register_pos_embed_hooks()
        # Register hooks for layers
        self.register_layer_hooks()
        # Register hooks for output head and norm layer
        self.register_head_hooks()

    def register_pos_embed_hooks(self):
        self.pos_embed.register_hook(log_gradient_norm("pos_embed"))
        self.temporal_pos_embedding.register_hook(log_gradient_norm("temporal_pos_embedding"))

    def register_layer_hooks(self):
        for idx, layer in enumerate(self.layers):
            # Register hooks for the layer's parameters
            for name, param in layer.named_parameters():
                param.register_hook(log_gradient_norm(f"layers.{idx}.{name}"))
            # Optionally, register hooks specifically for MambaSL
            # Uncomment the following lines if interested
            # mambasl = layer.mixer
            # for name, param in mambasl.named_parameters():
            #     param.register_hook(log_gradient_norm(f"layers.{idx}.mambasl.{name}"))

    def register_head_hooks(self):
        for name, param in self.head.named_parameters():
            param.register_hook(log_gradient_norm(f"head.{name}"))
        for name, param in self.norm_f.named_parameters():
            param.register_hook(log_gradient_norm(f"norm_f.{name}"))


    def create_valid_positions_mask(self, m, B, T, H, W, device):
        """
        Args:
            m: List of lists containing metadata for each frame in the batch.
              Shape: [B, T], where each element is a dict {label: [(x, y), ...]}
            B: Batch size
            T: Number of frames
            H: Height in patches
            W: Width in patches
        Returns:
            valid_positions: Boolean tensor of shape (B * T, H * W)
        """
        valid_positions = torch.zeros(B * T, H * W, dtype=torch.bool, device=device)
        for b in range(B):
            for t in range(T):
                index = b * T + t  # Flattened index
                frame_metadata = m[b][t]  # Dictionary {label: [(x, y), ...]}
                if 5 in frame_metadata:
                    coords = frame_metadata[5]  # List of (x, y) tuples
                    for x_coord, y_coord in coords:
                        # Ensure coordinates are within bounds
                        if 0 <= x_coord < W and 0 <= y_coord < H:
                            idx = y_coord * W + x_coord  # Flattened index for (H, W)
                            valid_positions[index, idx] = True

                num_valid = valid_positions[index].sum().item()
                logging.debug(f'Batch {b}, Frame {t}: {num_valid} valid positions')

        return valid_positions

    def select_patches_with_masks(self, x, valid_positions):
        """
        Prepare video sequence with masked patches while preserving spatiotemporal ordering.
        Args:
            x: Input tensor after patch embedding (B * T, N_patches, C)
            valid_positions: Boolean mask (B * T, N_patches)
        Returns:
            x_with_gaps: (B * T, N_patches, C) sequence with gaps
            valid_positions_seq: (B * T, N_patches) boolean mask
        """
        # Create sequences with gap tokens (set to zero)
        gap_value = 0.0  # Set to zero to ensure D * x_t is zero at invalid positions
        x_with_gaps = x.clone()
        x_with_gaps[~valid_positions] = gap_value
        valid_positions_seq = valid_positions
        return x_with_gaps, valid_positions_seq
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, m, inference_params=None):
        logging.debug(f'Input x shape: {x.shape}')
        x = self.patch_embed(x)  # x shape: (B, C, T, H, W)
        logging.debug(f'After patch embedding x shape: {x.shape}')
        B, C, T, H, W = x.shape
        logging.debug(f'Batch size: {B}, Channels: {C}, Frames: {T}, Height: {H}, Width: {W}')

        # Create valid_positions mask from m
        valid_positions = self.create_valid_positions_mask(m, B, T, H, W, x.device)  # (B * T, H * W)
        logging.debug(f'valid_positions shape: {valid_positions.shape}')
        logging.debug(f'Number of valid positions: {valid_positions.sum().item()}')

        # Reshape x to (B * T, H * W, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        logging.debug(f'Reshaped x shape: {x.shape}')

        # Prepare sequences with gaps
        x_with_gaps, valid_positions_seq = self.select_patches_with_masks(x, valid_positions)  # x_with_gaps: (B * T, H * W, C), valid_positions_seq: (B * T, H * W)
        logging.debug(f'x_with_gaps shape: {x_with_gaps.shape}')
        logging.debug(f'valid_positions_seq shape: {valid_positions_seq.shape}')

        # Prepend cls_token
        cls_token = self.cls_token.expand(B * T, -1, -1)  # shape: (B * T, 1, C)
        x_with_gaps = torch.cat((cls_token, x_with_gaps), dim=1)  # shape: (B * T, 1 + H * W, C)
        valid_positions_seq = torch.cat(
            (torch.ones(B * T, 1, dtype=torch.bool, device=x.device), valid_positions_seq), dim=1
        )  # shape: (B * T, 1 + H * W)

        logging.debug(f'After adding cls_token, x_with_gaps shape: {x_with_gaps.shape}')

        # Add spatial pos_embed
        x_with_gaps = x_with_gaps + self.pos_embed[:, :1 + H * W, :]  # pos_embed shape: (1, 1 + H * W, C)

        logging.debug(f'Added pos_embed, x_with_gaps shape: {x_with_gaps.shape}')

        # Reshape x to (B, T, 1 + H * W, C)
        x_with_gaps = x_with_gaps.view(B, T, 1 + H * W, C)
        valid_positions_seq = valid_positions_seq.view(B, T, 1 + H * W)

        logging.debug(f'After reshaping, x_with_gaps shape: {x_with_gaps.shape}')

        # Add temporal positional embeddings to CLS token
        x_with_gaps[:, :, 0, :] = x_with_gaps[:, :, 0, :] + self.temporal_pos_embedding[:, :T, :].transpose(1, 0)  # (B, T, C)

        logging.debug(f'Added temporal_pos_embedding to CLS tokens')

        # Reshape back to (B, T * (1 + H * W), C)
        x_with_gaps = x_with_gaps.view(B, T * (1 + H * W), C)
        valid_positions_seq = valid_positions_seq.view(B, T * (1 + H * W))

        logging.debug(f'Final x_with_gaps shape before passing to Mamba layers: {x_with_gaps.shape}')
        logging.debug(f'Final valid_positions_seq shape: {valid_positions_seq.shape}')

        x = self.pos_drop(x_with_gaps)

        # Mamba implementation
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):

            logging.debug(f'Passing through layer {idx}')

            hidden_states, residual = layer(
                hidden_states,
                valid_positions=valid_positions_seq,
                inference_params=inference_params,
            )
      
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token

        logging.debug(f'Final CLS token shape: {hidden_states[:, 0, :].shape}')

        return hidden_states[:, 0, :]

    def forward(self, x, m, inference_params=None):
        x = self.forward_features(x, m, inference_params)
        x = self.head(self.head_drop(x))
        return x
    

def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=False, 
        residual_in_fp32=True, 
        fused_add_norm=False, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=1)
    parser.add_argument('--orig_t_size', type=int, default=8)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    parser.add_argument('--use_learnable_pos_emb', action='store_true')
    parser.set_defaults(use_learnable_pos_emb=False)

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--checkpoint_path', default='checkpoints/videomamba_m16_k400_f16_res224.pth', help='finetune from checkpoint')
    parser.add_argument('--delete_head', action='store_true', help='whether delete head')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', default=0, type=int,
                        help='number of layers for using checkpoint')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--dataset', default='celebdf', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='you_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--loader_classes', default=18, type=int,
                        help='number of the data loader classification types')
    parser.add_argument('--pretraining', default='COCO', type=str,
                        help='semantic dataloader pretraining type')
    
    parser.add_argument('--test_list_path', default='celebdf_dataset/List_of_testing_videos.txt', type=str,
                        help='semantic dataloader test split file')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='semantic dataloader opt size')
    parser.add_argument('--deeplabv2', default=True, type=bool,
                        help='semantic dataloader pretraining type')
    parser.add_argument('--semantic_loading', default=False, type=bool,
                        help='semantic dataloader pretraining type')
    parser.add_argument('--checkpoint_dir', default='checkpoints/Deeplab', type=str,
                        help='semantic dataloader backbone checkpoints')

    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', action='store_true',
                        help='whether use decord to load video, otherwise load image')
    parser.add_argument('--no_use_decord', action='store_false', dest='use_decord')
    parser.set_defaults(use_decord=True)
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--trimmed', type=int, default=60)
    parser.add_argument('--time_stride', type=int, default=16)
    parser.add_argument('--output_dir', default='checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)
    parser.add_argument('--no_amp', action='store_true')
    parser.set_defaults(no_amp=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--bf16', default=False, action='store_true')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def class_pixel_loader(args, mode):
    return CelebDFDataSet(root_path=args.data_path, test_list_path=args.test_list_path, mode=mode, clip_len=args.num_frames, frame_sample_rate=args.sampling_rate, crop_size=args.crop_size, semantic_loading=args.semantic_loading, args=args)


def freeze_layers(model, num_layers_to_freeze=None):
    """
    Freeze layers of the model starting from the bottom
    Args:
        model: The model to freeze
        num_layers_to_freeze: Number of layers to freeze from bottom. If None, freeze all except head
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    # Get model type and determine architecture
    if 'videomamba' in model.__class__.__name__.lower():
        blocks = model.blocks if hasattr(model, 'blocks') else []
        total_layers = len(blocks)
    elif hasattr(model, 'get_num_layers'):
        total_layers = model.get_num_layers()
    else:
        raise ValueError("Unknown model architecture - cannot determine number of layers")

    # Validate num_layers_to_freeze
    if num_layers_to_freeze is not None:
        if not isinstance(num_layers_to_freeze, int) or num_layers_to_freeze < 0:
            raise ValueError("num_layers_to_freeze must be a non-negative integer")
        if num_layers_to_freeze > total_layers:
            print(f"Warning: num_layers_to_freeze ({num_layers_to_freeze}) is greater than total layers ({total_layers})")
            num_layers_to_freeze = total_layers

    # For Stage 1 (num_layers_to_freeze is None), keep everything frozen except head
    if num_layers_to_freeze is None:
        # Only unfreeze head
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        print("Stage 1: Froze all layers except head")
        return

    # For other stages, unfreeze layers from top to bottom
    frozen_layers = set()
    unfrozen_layers = set()
    
    for name, param in model.named_parameters():
        # Always unfreeze head
        if 'head' in name:
            param.requires_grad = True
            continue
            
        # Keep patch embedding frozen for all stages
        if any(x in name for x in ['patch_embed', 'pos_embed', 'cls_token']):
            param.requires_grad = False
            frozen_layers.add('embedding')
            continue

        # Handle different model architectures
        layer_num = None
        if 'videomamba' in model.__class__.__name__.lower():
            if 'blocks.' in name:
                layer_num = int(name.split('blocks.')[1].split('.')[0])
        elif 'deit' in model.__class__.__name__.lower() or 'vit' in model.__class__.__name__.lower():
            if 'block.' in name:
                layer_num = int(name.split('block.')[1].split('.')[0])
        else:
            if name.split('.')[0] in ['blocks', 'block']:
                layer_num = int(name.split('.')[1])

        if layer_num is not None:
            if layer_num < num_layers_to_freeze:
                param.requires_grad = False
                frozen_layers.add(layer_num)
            else:
                param.requires_grad = True
                unfrozen_layers.add(layer_num)

    print(f"Frozen layers: {sorted(frozen_layers)}")
    print(f"Unfrozen layers: {sorted(unfrozen_layers)}")
    print(f"Total frozen layers: {len(frozen_layers)}")
    print(f"Total unfrozen layers: {len(unfrozen_layers)}")


def verify_layer_freezing(model):
    """
    Verify which layers are frozen/unfrozen
    """
    frozen = []
    unfrozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen.append(name)
        else:
            frozen.append(name)
            
    print("\nFrozen parameters:")
    for name in frozen:
        print(f"  {name}")
    print("\nUnfrozen parameters:")
    for name in unfrozen:
        print(f"  {name}")
    
    return len(frozen), len(unfrozen)


def get_stage_config(stage, args):
    """
    Get configuration for different training stages
    Args:
        stage: Training stage (1, 2, or 3)
        args: Original training arguments
    Returns:
        Modified configuration for the stage
    """
    if stage not in [1, 2, 3]:
        raise ValueError("Stage must be 1, 2, or 3")
        
    config = copy.deepcopy(args)
    
    # Stage-specific configurations
    stage_configs = {
        1: {  # Feature extraction
            'epochs': min(5, args.epochs),
            'lr_factor': 0.1,
            'layer_decay': 1.0,
            'weight_decay_factor': 0.1,
            'frozen_layers': None  # Freeze all except head
        },
        2: {  # Partial fine-tuning
            'epochs': min(10, args.epochs),
            'lr_factor': 0.5,
            'layer_decay': args.layer_decay,
            'weight_decay_factor': 1.0,
            'frozen_layers': args.num_layers // 2 if hasattr(args, 'num_layers') else 6
        },
        3: {  # Full fine-tuning
            'epochs': args.epochs,
            'lr_factor': 1.0,
            'layer_decay': args.layer_decay,
            'weight_decay_factor': 1.0,
            'frozen_layers': 0
        }
    }
    
    stage_config = stage_configs[stage]
    
    # Apply configurations
    config.epochs = stage_config['epochs']
    config.lr = args.lr * stage_config['lr_factor']
    config.layer_decay = stage_config['layer_decay']
    config.weight_decay = args.weight_decay * stage_config['weight_decay_factor']
    config.frozen_layers = stage_config['frozen_layers']
    
    # Adjust warmup for each stage
    config.warmup_epochs = max(1, args.warmup_epochs // (4-stage))
    
    return config


def reset_optimizer_and_scheduler(model, stage_config, num_steps_per_epoch, assigner=None):
    """
    Create fresh optimizer and scheduler for each stage
    """
    optimizer = create_optimizer(
        stage_config, model,
        skip_list=model.no_weight_decay() if hasattr(model, 'no_weight_decay') else [],
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    
    lr_schedule = utils.cosine_scheduler(
        stage_config.lr, 
        stage_config.min_lr, 
        stage_config.epochs,
        num_steps_per_epoch,
        warmup_epochs=stage_config.warmup_epochs,
        start_warmup_value=stage_config.warmup_lr
    )
    
    return optimizer, lr_schedule


def cleanup_stage(model_ema=None):
    """
    Cleanup resources after each stage
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if model_ema is not None:
        model_ema.restore_backup()
        
    gc.collect()


def staged_training(model, args, data_loaders, device, criterion, amp_autocast, mixup_fn=None, 
                   assigner=None, model_ema=None, loss_scaler=None, log_writer=None):
    """
    Enhanced staged training with proper error handling and resource management
    """
    if not all(k in data_loaders for k in ['train', 'val', 'test']):
        raise ValueError("data_loaders must contain 'train', 'val', and 'test' keys")
    
    training_stats = []
    global_best_acc = 0.0
    
    try:
        for stage in range(1, 4):
            print(f"\n{'='*20} Starting Stage {stage} {'='*20}")
            stage_config = get_stage_config(stage, args)
            
            # Apply layer freezing with verification
            freeze_layers(model, stage_config.frozen_layers)
            frozen_count, unfrozen_count = verify_layer_freezing(model)
            print(f"Stage {stage}: {frozen_count} frozen parameters, {unfrozen_count} unfrozen parameters")

            total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
            num_training_steps_per_epoch = len(data_loaders['train'].dataset) // total_batch_size
            print("STEPS PER EPOCH: ", num_training_steps_per_epoch)
            
            # Reset optimizer and scheduler for new stage
            optimizer, lr_schedule = reset_optimizer_and_scheduler(model, stage_config, num_training_steps_per_epoch, assigner)
            
            # Initialize early stopping
            patience = max(2, stage_config.epochs // 10)  # 10% of stage epochs
            best_acc = 0.0
            patience_counter = 0
            
            stage_stats = {'stage': stage, 'epochs': [], 'best_acc': 0.0}
            
            for epoch in range(stage_config.epochs):
                if args.distributed:
                    data_loaders['train'].sampler.set_epoch(epoch)
                    torch.distributed.barrier()
                
                # Clear gradients before training
                model.zero_grad()
                if model_ema is not None:
                    model_ema.zero_grad()
                
                try:
                    # Training
                    if args.distributed:
                        train_stats = train_one_epoch(
                            model, criterion, data_loaders['train'], optimizer,
                            device, epoch, loss_scaler, amp_autocast, stage_config.clip_grad,
                            model_ema=model_ema, mixup_fn=mixup_fn,
                            log_writer=log_writer, start_steps=epoch * len(data_loaders['train']),
                            lr_schedule_values=lr_schedule,
                            num_training_steps_per_epoch=num_training_steps_per_epoch,
                            update_freq=stage_config.update_freq,
                            no_amp=args.no_amp,
                            bf16=args.bf16
                        )
                    else:
                        train_stats = train_one_epoch_no_dist(
                            model, criterion, data_loaders['train'], optimizer,
                            device, epoch, loss_scaler, amp_autocast, stage_config.clip_grad,
                            model_ema=model_ema, mixup_fn=mixup_fn,
                            log_writer=log_writer, start_steps=epoch * len(data_loaders['train']),
                            lr_schedule_values=lr_schedule,
                            num_training_steps_per_epoch=num_training_steps_per_epoch,
                            update_freq=stage_config.update_freq,
                            no_amp=args.no_amp,
                            bf16=args.bf16
                        )
                        
                    # Validation
                    if data_loaders['val'] is not None:
                        if args.distributed:
                            test_stats = validation_one_epoch(
                                data_loaders['val'], model, device, amp_autocast,
                                ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
                                maxk=5 if args.nb_classes >= 5 else 2
                            )
                        
                            torch.distributed.barrier()
                            # Gather accuracy from all processes
                            acc1_tensor = torch.tensor(test_stats['acc1']).cuda()
                            torch.distributed.all_reduce(acc1_tensor)
                            test_stats['acc1'] = acc1_tensor.item() / torch.distributed.get_world_size()

                        else:
                            test_stats = validation_one_epoch_no_dist(
                                data_loaders['val'], model, device, amp_autocast,
                                ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
                                maxk=5 if args.nb_classes >= 5 else 2
                            )
                        
                        # Update best accuracy and save checkpoint
                        is_best = test_stats['acc1'] > best_acc
                        if is_best:
                            best_acc = test_stats['acc1']
                            stage_stats['best_acc'] = best_acc
                            patience_counter = 0
                            if best_acc > global_best_acc:
                                global_best_acc = best_acc

                                with open("val_debug.txt", 'a', encoding='utf-8') as file:
                                    file.write(str(test_stats))

                        else:
                            patience_counter += 1
                        
                        # Early stopping check
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break
                    
                except RuntimeError as e:
                    print(f"Error during epoch: {str(e)}")
                    if "out of memory" in str(e):
                        cleanup_stage(model_ema)
                    raise e
                
                # Log statistics
                if not args.distributed or utils.is_main_process():
                    epoch_stats = {
                        'epoch': epoch,
                        'train_loss': train_stats['loss'],
                        'val_acc1': test_stats['acc1'] if data_loaders['val'] is not None else None,
                        'lr': optimizer.param_groups[0]['lr']
                    }
                    stage_stats['epochs'].append(epoch_stats)
                    
                    print(f"Stage {stage}, Epoch {epoch}: "
                          f"train_loss: {train_stats['loss']:.3f}, "
                          f"val_acc1: {test_stats['acc1']:.2f}% "
                          f"lr: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Cleanup after stage
            cleanup_stage(model_ema)
            training_stats.append(stage_stats)
            
        return training_stats, global_best_acc
        
    except Exception as e:
        print(f"Error during staged training: {str(e)}")
        cleanup_stage(model_ema)
        raise


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    if 'videomamba' in args.model:
        model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.nb_classes,
            fc_drop_rate=args.fc_drop_rate,
            drop_path_rate=args.drop_path,
            kernel_size=args.tubelet_size,
            num_frames=args.num_frames,
            use_checkpoint=args.use_checkpoint,
            checkpoint_num=args.checkpoint_num,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            use_learnable_pos_emb=args.use_learnable_pos_emb,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_checkpoint=args.use_checkpoint,
            checkpoint_num=args.checkpoint_num,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset_train = class_pixel_loader(args, "train")

    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = class_pixel_loader(args, "validation")
    dataset_test = class_pixel_loader(args, "test")
    

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = def_collate

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True,
        multiprocessing_context='spawn'
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=collate_func,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=collate_func,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    print("Load ckpt from %s" % args.checkpoint_path)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    if 'head.weight' in checkpoint_model.keys():
        if args.delete_head:
            print("Removing head from pretrained checkpoint")
            del checkpoint_model['head.weight']
            del checkpoint_model['head.bias']
        elif checkpoint_model['head.weight'].shape[0] == 710:
            if args.nb_classes == 400:
                checkpoint_model['head.weight'] = checkpoint_model['head.weight'][:args.nb_classes]
                checkpoint_model['head.bias'] = checkpoint_model['head.bias'][:args.nb_classes]
            elif args.nb_classes in [600, 700]:
                # download from https://drive.google.com/drive/folders/17cJd2qopv-pEG8NSghPFjZo1UUZ6NLVm
                map_path = f'k710/label_mixto{args.nb_classes}.json'
                print(f'Load label map from {map_path}')
                with open(map_path) as f:
                    label_map = json.load(f)
                checkpoint_model['head.weight'] = checkpoint_model['head.weight'][label_map]
                checkpoint_model['head.bias'] = checkpoint_model['head.bias'][label_map]
                
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # interpolate position embedding
    if 'deit' in args.model or 'videomamba' in args.model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B, H, W, C -> B, C, H, W
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # B, C, H, W -> B, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size * new_size, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
        # Handle temporal_pos_embedding interpolation
        print("Interpolating temporal_pos_embedding")
        temporal_pos_embed = checkpoint_model['temporal_pos_embedding']  # [1, old_num_positions, embed_dim]

        with open('debug.txt', 'a') as file:
            file.write(f"Original temporal_pos_embedding shape: {temporal_pos_embed.shape}")

        # we use 8 frames for pretraining
        orig_t_size = args.orig_t_size // model.patch_embed.tubelet_size
        new_t_size = args.num_frames // model.patch_embed.tubelet_size
        # height (== width) for the checkpoint position embedding
        if orig_t_size != new_t_size:
            print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
            temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
            temporal_pos_embed = torch.nn.functional.interpolate(
                temporal_pos_embed, size=(new_t_size,), mode='linear', align_corners=False
            )
            temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
            checkpoint_model['temporal_pos_embedding'] = temporal_pos_embed

        with open('debug.txt', 'a') as file:
            file.write(f"Interpolated temporal_pos_embedding shape: {temporal_pos_embed.shape}")

    elif 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # we use 8 frames for pretraining
        orig_t_size = args.orig_t_size // model.patch_embed.tubelet_size
        new_t_size = args.num_frames // model.patch_embed.tubelet_size
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // new_t_size) ** 0.5)
        
        if orig_t_size != new_t_size:
            print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
            tmp_pos_embed = pos_embed_checkpoint.view(1, orig_t_size, -1, embedding_size)
            tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            tmp_pos_embed = torch.nn.functional.interpolate(tmp_pos_embed, size=new_t_size, mode='linear')
            tmp_pos_embed = tmp_pos_embed.view(1, -1, embedding_size, new_t_size)
            tmp_pos_embed = tmp_pos_embed.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            checkpoint_model['pos_embed'] = tmp_pos_embed
            pos_embed_checkpoint = tmp_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size * args.num_sample / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Repeated sample = %d" % args.num_sample)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    amp_autocast = contextlib.nullcontext()
    loss_scaler = "none"
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)

        if not args.no_amp:
            print(f"Use bf16: {args.bf16}")
            dtype = torch.bfloat16 if args.bf16 else torch.float16
            amp_autocast = torch.cuda.amp.autocast(dtype=dtype)
            loss_scaler = NativeScaler()

    """print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps,
    )"""

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        print("Using SoftTargetCrossEntropy")
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        print("Using LabelSmoothingCrossEntropy")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')

        if args.distributed:
            test_stats = final_test(
                data_loader_test, model, device, preds_file, amp_autocast,
                ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
                maxk=5 if args.nb_classes >= 5 else 1
            )
            torch.distributed.barrier()

        else:
            test_stats = final_test_no_dist(
                data_loader_test, model, device, preds_file, amp_autocast,
                ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
                maxk=5 if args.nb_classes >= 5 else 1
            )

        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)
        

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    if not args.eval:
        data_loaders = {
            'train': data_loader_train,
            'val': data_loader_val,
            'test': data_loader_test
        }
        
        try:
            stats, best_accuracy = staged_training(
                model=model,
                args=args,
                data_loaders=data_loaders,
                device=device,
                criterion=criterion,
                amp_autocast=amp_autocast,
                mixup_fn=mixup_fn,
                assigner=assigner,
                model_ema=model_ema,
                loss_scaler=loss_scaler,
                log_writer=log_writer
            )
            
            if max_accuracy < best_accuracy:
                max_accuracy = best_accuracy
            
            # Save and log final results
            if args.output_dir and (not args.distributed or utils.is_main_process()):
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    for stage_stat in stats:
                        f.write(json.dumps(stage_stat) + "\n")
                    f.write(json.dumps({'final_best_accuracy': best_accuracy}) + "\n")
                
                if log_writer is not None:
                    log_writer.flush()
        
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

    
    print("MAX ACCURACY: ", max_accuracy)
    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    if args.test_best:
        print("Auto testing the best model")
        args.eval = True
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.distributed:
        test_stats = final_test(
            data_loader_test, model, device, preds_file, amp_autocast,
            ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
            maxk=5 if args.nb_classes >= 5 else 1
        )
        torch.distributed.barrier()

    else:
        test_stats = final_test_no_dist(
            data_loader_test, model, device, preds_file, amp_autocast,
            ds=args.enable_deepspeed, no_amp=args.no_amp, bf16=args.bf16,
            maxk=5 if args.nb_classes >= 5 else 1
        )

    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)