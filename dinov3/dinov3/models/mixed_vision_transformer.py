# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dinov3.layers import MixedSelfAttentionBlock

from .vision_transformer import DinoVisionTransformer


class MixedDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, *, num_vanilla_heads: int = 0, **kwargs):
        self._num_vanilla_heads = num_vanilla_heads
        super().__init__(**kwargs)

    def _make_block(self, **kwargs):
        return MixedSelfAttentionBlock(num_vanilla_heads=self._num_vanilla_heads, **kwargs)


def vit_small_mixed(patch_size=16, **kwargs):
    model = MixedDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model
