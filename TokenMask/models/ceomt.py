# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scale_block import ScaleBlock


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        mask_upsample_mode: str = "feature",  # {"feature", "logit", "none"}
        mask_out_stride: Optional[int] = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = int(num_q)
        self.num_blocks = int(num_blocks)
        self.masked_attn_enabled = bool(masked_attn_enabled)
        self.mask_upsample_mode = str(mask_upsample_mode)
        self.mask_out_stride = mask_out_stride

        self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))

        self.q = nn.Embedding(self.num_q, int(self.encoder.backbone.embed_dim))

        self.class_head = nn.Linear(int(self.encoder.backbone.embed_dim), int(num_classes) + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(int(self.encoder.backbone.embed_dim), int(self.encoder.backbone.embed_dim)),
            nn.GELU(),
            nn.Linear(int(self.encoder.backbone.embed_dim), int(self.encoder.backbone.embed_dim)),
            nn.GELU(),
            nn.Linear(int(self.encoder.backbone.embed_dim), int(self.encoder.backbone.embed_dim)),
        )

        # Upscaling strategy:
        # - "feature" (default, original): upscale feature maps (C channels) before mask dot-product (expensive)
        # - "logit": compute mask logits at patch resolution, then (optionally) upsample logits (Q channels)
        # - "none": no upscaling at all; masks are returned at patch resolution
        if self.mask_upsample_mode == "feature":
            patch_size = self.encoder.backbone.patch_embed.patch_size
            max_patch_size = max(int(patch_size[0]), int(patch_size[1]))
            # Original code upsamples from patch stride (e.g., 16) to stride 4 => log2(16)-2 = 2 blocks
            num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
            self.upscale = nn.Sequential(
                *[ScaleBlock(int(self.encoder.backbone.embed_dim)) for _ in range(num_upscale)]
            )
        else:
            self.upscale = nn.Identity()

    # -------------------------
    # Helpers
    # -------------------------

    def _infer_grid_size_from_tokens(self, n_tokens: int) -> Tuple[int, int]:
        """
        Robustly infer a (H, W) grid from the number of patch tokens.
        Prefer the backbone's stored grid_size if it matches; otherwise choose a close factorization.
        This fixes patch14 + non-multiple-of-14 sizes like 640x640 where N may be 45*44.
        """
        # Use stored grid_size when consistent.
        if hasattr(self.encoder.backbone, "patch_embed") and hasattr(self.encoder.backbone.patch_embed, "grid_size"):
            gs = self.encoder.backbone.patch_embed.grid_size
            try:
                gh, gw = int(gs[0]), int(gs[1])
                if gh * gw == n_tokens:
                    return gh, gw
            except Exception:
                pass

        # Try to infer from the input image size if available.
        # (EoMT.forward sees the original x; but here we don't have it. So we do factorization.)
        # Choose factorization with minimal aspect ratio distortion (closest to square).
        s = int(math.isqrt(n_tokens))
        for d in range(s, 0, -1):
            if n_tokens % d == 0:
                h = d
                w = n_tokens // d
                return h, w

        # Fallback (should never happen)
        return 1, n_tokens

    def _tokens_to_feature_map(self, feats_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens (B, N, C) to feature map (B, C, H, W) with robust grid size.
        """
        B, N, C = feats_tokens.shape
        gh, gw = self._infer_grid_size_from_tokens(N)
        # reshape to (B, C, H, W)
        return feats_tokens.transpose(1, 2).reshape(B, C, gh, gw)

    # -------------------------
    # Core
    # -------------------------

    def _predict(self, x: torch.Tensor):
        """Predict mask logits and class logits.

        Depending on `mask_upsample_mode`:
          - feature: upscale feature maps before dot-product (original behavior)
          - logit:  dot-product at patch resolution, then upsample logits (optional)
          - none:   no upscaling at all (patch-resolution masks)
        """
        q = x[:, : self.num_q, :]
        class_logits = self.class_head(q)

        feats = x[:, self.num_q + int(self.encoder.backbone.num_prefix_tokens) :, :]  # (B, N, C)
        feats = self._tokens_to_feature_map(feats)  # (B, C, H, W)

        if self.mask_upsample_mode == "feature":
            mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(feats))
            return mask_logits, class_logits

        # Compute logits at patch resolution (cheap).
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), feats)

        if self.mask_upsample_mode == "logit":
            # Upsample logits from patch stride (e.g., 16/14) to `mask_out_stride` (e.g., 4).
            if self.mask_out_stride is None:
                raise ValueError("mask_out_stride must be set when mask_upsample_mode='logit'")

            patch_size = self.encoder.backbone.patch_embed.patch_size
            patch = max(int(patch_size[0]), int(patch_size[1]))
            if patch % int(self.mask_out_stride) != 0:
                raise ValueError(
                    f"patch_size ({patch}) must be divisible by mask_out_stride ({self.mask_out_stride})"
                )

            scale = patch // int(self.mask_out_stride)
            if scale > 1:
                mask_logits = F.interpolate(
                    mask_logits,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )

        # mask_upsample_mode == "none" => return patch-resolution logits
        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask: torch.Tensor, prob: float):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device) > prob
            )
            attn_mask[:, : self.num_q, self.num_q + int(self.encoder.backbone.num_prefix_tokens) :][
                random_queries
            ] = True
        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )

        # We want the attention mask aligned to the patch-token grid (whatever it is).
        # Infer it from the number of patch tokens currently in x.
        # x contains: [queries] + [prefix tokens] + [patch tokens]
        n_patch = x.shape[1] - self.num_q - int(self.encoder.backbone.num_prefix_tokens)
        gh, gw = self._infer_grid_size_from_tokens(int(n_patch))

        if mask_logits.shape[-2:] == (gh, gw):
            interpolated = mask_logits
        else:
            interpolated = F.interpolate(
                mask_logits,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            )

        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + int(self.encoder.backbone.num_prefix_tokens) :,
        ] = (interpolated > 0)

        idx = i - len(self.encoder.backbone.blocks) + self.num_blocks
        attn_mask = self._disable_attn_mask(attn_mask, float(self.attn_mask_probs[idx]))
        return attn_mask

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        # Apply positional embedding if exposed by the backbone
        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        # If the backbone exposes no transformer blocks (e.g., AFFormer wrapper),
        # we must inject queries here, otherwise _predict() will treat patch tokens as queries.
        if len(self.encoder.backbone.blocks) == 0:
            x = torch.cat(
                (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
            )

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if self.masked_attn_enabled and i >= len(self.encoder.backbone.blocks) - self.num_blocks:
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention

            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)

            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))

            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (mask_logits_per_layer, class_logits_per_layer)
