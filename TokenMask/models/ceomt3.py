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
    """
    EoMT with configurable mask prediction + upsampling strategies.

    Key knobs:
      - mask_upsample_mode:
          * "feature": upscale feature maps (C channels) before dot-product (original, expensive)
          * "logit":   compute mask logits at patch resolution, then (optionally) upsample logits (Q channels)
          * "none":    no upscaling; masks returned at patch resolution
      - mask_pred_mode:
          * "feature_map": mask logits from (B,C,H,W) feature map (original EoMT head)
          * "token":      mask logits directly in token space: (B,Q,N) = q · token^T (no feature map needed)
      - lowres_mask_for_attn:
          * If True: when masked attention is enabled, build attention masks from patch-resolution logits
            (avoids upsample->downsample ping-pong). Final outputs can still be upsampled.
      - cosine_mask_logits:
          * If True (token mode): use cosine similarity for mask logits (often stabilizes training).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        mask_upsample_mode: str = "feature",  # {"feature", "logit", "none"}
        mask_out_stride: Optional[int] = 4,
        mask_pred_mode: str = "feature_map",  # {"feature_map", "token"}
        lowres_mask_for_attn: bool = True,
        cosine_mask_logits: bool = False,
        token_mask_softmax: bool = False,  # optional experiment: apply softmax over tokens (usually False)
        token_mask_temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = int(num_q)
        self.num_blocks = int(num_blocks)
        self.masked_attn_enabled = bool(masked_attn_enabled)
        self.mask_upsample_mode = str(mask_upsample_mode)
        self.mask_out_stride = mask_out_stride

        self.mask_pred_mode = str(mask_pred_mode)
        if self.mask_pred_mode not in {"feature_map", "token"}:
            raise ValueError(f"mask_pred_mode must be one of {{'feature_map','token'}}, got {self.mask_pred_mode}")

        self.lowres_mask_for_attn = bool(lowres_mask_for_attn)
        self.cosine_mask_logits = bool(cosine_mask_logits)
        self.token_mask_softmax = bool(token_mask_softmax)
        self.token_mask_temperature = float(token_mask_temperature)

        self.register_buffer("attn_mask_probs", torch.ones(self.num_blocks))

        embed_dim = int(self.encoder.backbone.embed_dim)

        self.q = nn.Embedding(self.num_q, embed_dim)

        self.class_head = nn.Linear(embed_dim, int(num_classes) + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Useful scale for attention-like scores (token mode)
        self.register_buffer(
            "mask_logit_scale",
            torch.tensor(1.0 / math.sqrt(float(embed_dim))),
            persistent=False,
        )

        # Upscaling strategy for "feature" upsampling
        # (only needed when we upscale feature maps before dot-product)
        if self.mask_upsample_mode == "feature" and self.mask_pred_mode == "feature_map":
            patch_size = self.encoder.backbone.patch_embed.patch_size
            max_patch_size = max(int(patch_size[0]), int(patch_size[1]))
            num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
            self.upscale = nn.Sequential(*[ScaleBlock(embed_dim) for _ in range(num_upscale)])
        else:
            self.upscale = nn.Identity()

    # -------------------------
    # Helpers
    # -------------------------

    def _infer_grid_size_from_tokens(self, n_tokens: int) -> Tuple[int, int]:
        """
        Robustly infer a (H, W) grid from the number of patch tokens.
        Prefer the backbone's stored grid_size if it matches; otherwise choose a close factorization.
        Fixes patch14 + non-multiple-of-14 sizes like 640x640 where N may be 45*44.
        """
        if hasattr(self.encoder.backbone, "patch_embed") and hasattr(self.encoder.backbone.patch_embed, "grid_size"):
            gs = self.encoder.backbone.patch_embed.grid_size
            try:
                gh, gw = int(gs[0]), int(gs[1])
                if gh * gw == n_tokens:
                    return gh, gw
            except Exception:
                pass

        s = int(math.isqrt(n_tokens))
        for d in range(s, 0, -1):
            if n_tokens % d == 0:
                h = d
                w = n_tokens // d
                return h, w

        return 1, n_tokens

    def _tokens_to_feature_map(self, feats_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens (B, N, C) to feature map (B, C, H, W) with robust grid size.
        """
        B, N, C = feats_tokens.shape
        gh, gw = self._infer_grid_size_from_tokens(N)
        return feats_tokens.transpose(1, 2).reshape(B, C, gh, gw)

    def _get_patch_tokens(self, x_tokens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x_tokens: (B, N_total, C) = [queries] + [prefix tokens] + [patch tokens]
        returns: patch_tokens (B, N, C) and grid (gh, gw)
        """
        patch_tokens = x_tokens[:, self.num_q + int(self.encoder.backbone.num_prefix_tokens) :, :]  # (B, N, C)
        B, N, C = patch_tokens.shape
        gh, gw = self._infer_grid_size_from_tokens(N)
        return patch_tokens, gh, gw

    def _upsample_logits_if_needed(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """
        Upsample mask logits from patch stride to mask_out_stride if mask_upsample_mode == "logit".
        If mask_upsample_mode == "none", returns as-is.
        """
        if self.mask_upsample_mode != "logit":
            return mask_logits

        if self.mask_out_stride is None:
            raise ValueError("mask_out_stride must be set when mask_upsample_mode='logit'")

        patch_size = self.encoder.backbone.patch_embed.patch_size
        patch = max(int(patch_size[0]), int(patch_size[1]))
        if patch % int(self.mask_out_stride) != 0:
            raise ValueError(f"patch_size ({patch}) must be divisible by mask_out_stride ({self.mask_out_stride})")

        scale = patch // int(self.mask_out_stride)
        if scale > 1:
            mask_logits = F.interpolate(
                mask_logits,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            )
        return mask_logits

    # -------------------------
    # Core prediction
    # -------------------------

    def _predict_feature_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          - mask_logits_out: final mask logits (maybe upsampled depending on mask_upsample_mode)
          - class_logits
          - mask_logits_patch: patch-resolution logits (for attn masking if desired)
        """
        q = x[:, : self.num_q, :]
        class_logits = self.class_head(q)

        feats_tokens = x[:, self.num_q + int(self.encoder.backbone.num_prefix_tokens) :, :]  # (B, N, C)
        feats = self._tokens_to_feature_map(feats_tokens)  # (B, C, H, W) with H,W on patch grid

        # Patch-resolution logits (always available)
        mask_logits_patch = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), feats)

        # Output strategy
        if self.mask_upsample_mode == "feature":
            # expensive path: upscale feature maps before dot-product
            mask_logits_out = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(feats))
            return mask_logits_out, class_logits, mask_logits_patch

        # "logit" or "none": compute at patch, then optional upsample logits
        mask_logits_out = mask_logits_patch
        mask_logits_out = self._upsample_logits_if_needed(mask_logits_out)
        return mask_logits_out, class_logits, mask_logits_patch

    def _predict_token(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Token-space mask prediction:
          mask_logits_patch: (B,Q,gh,gw) from q·token^T, reshaped to patch grid.
        Returns:
          - mask_logits_out: final mask logits (maybe upsampled)
          - class_logits
          - mask_logits_patch: patch-resolution logits (for attn masking)
        """
        q = x[:, : self.num_q, :]
        class_logits = self.class_head(q)

        patch_tokens, gh, gw = self._get_patch_tokens(x)  # (B,N,C)

        q_mask = self.mask_head(q)  # (B,Q,C)

        if self.cosine_mask_logits:
            q_mask_n = F.normalize(q_mask, dim=-1)
            patch_n = F.normalize(patch_tokens, dim=-1)
            logits = torch.einsum("bqc,bnc->bqn", q_mask_n, patch_n)
        else:
            logits = torch.einsum("bqc,bnc->bqn", q_mask, patch_tokens) * self.mask_logit_scale

        # Optional "attention-like" softmax over tokens (rarely needed; keep False by default)
        if self.token_mask_softmax:
            t = max(1e-6, self.token_mask_temperature)
            logits = logits / t
            logits = F.softmax(logits, dim=-1)

        mask_logits_patch = logits.view(logits.size(0), logits.size(1), gh, gw)

        # Output strategy (feature upsample doesn't apply here; treat as "none"/"logit")
        mask_logits_out = mask_logits_patch
        mask_logits_out = self._upsample_logits_if_needed(mask_logits_out)
        return mask_logits_out, class_logits, mask_logits_patch

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unified predict:
          returns (mask_logits_out, class_logits, mask_logits_patch)
        """
        if self.mask_pred_mode == "token":
            # In token mode, "feature" upsample doesn't make sense; we still allow it but it behaves like "none/logit".
            return self._predict_token(x)
        return self._predict_feature_map(x)

    # -------------------------
    # Attention + masking
    # -------------------------

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask: torch.Tensor, prob: float):
        if prob < 1:
            random_queries = (torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device) > prob)
            attn_mask[:, : self.num_q, self.num_q + int(self.encoder.backbone.num_prefix_tokens) :][random_queries] = True
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

    def _attn_mask(self, x: torch.Tensor, mask_logits_for_attn: torch.Tensor, i: int):
        """
        Build boolean attention mask aligned to patch-token grid.
        mask_logits_for_attn can be patch-res or higher-res; we interpolate to patch grid.
        """
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )

        n_patch = x.shape[1] - self.num_q - int(self.encoder.backbone.num_prefix_tokens)
        gh, gw = self._infer_grid_size_from_tokens(int(n_patch))

        if mask_logits_for_attn.shape[-2:] == (gh, gw):
            interpolated = mask_logits_for_attn
        else:
            interpolated = F.interpolate(
                mask_logits_for_attn,
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

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        if len(self.encoder.backbone.blocks) == 0:
            x = torch.cat((self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat((self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)

            if self.masked_attn_enabled and i >= len(self.encoder.backbone.blocks) - self.num_blocks:
                mask_logits_out, class_logits, mask_logits_patch = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits_out)
                class_logits_per_layer.append(class_logits)

                # Which logits should drive the attention mask?
                if self.lowres_mask_for_attn:
                    attn_mask = self._attn_mask(x, mask_logits_patch, i)
                else:
                    attn_mask = self._attn_mask(x, mask_logits_out, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention

            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)

            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)
            else:
                x = x + attn_out

            mlp_out = block.mlp(block.norm2(x))

            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)
            else:
                x = x + mlp_out

        mask_logits_out, class_logits, _mask_logits_patch = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits_out)
        class_logits_per_layer.append(class_logits)

        return (mask_logits_per_layer, class_logits_per_layer)
