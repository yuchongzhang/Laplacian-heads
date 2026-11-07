"""DeiT-B with vanilla and Laplacian attention heads.

Adapted from timm. Copyright 2020 Ross Wightman.
"""
from typing import Optional, Type
import torch
from torch import nn
from timm.layers import MixedAttention, DropPath, LayerScale, Mlp, LayerNorm, PatchEmbed, trunc_normal_

class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_vanilla_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}

        self.norm1 = norm_layer(dim, **dd)
        assert num_vanilla_heads <= num_heads
        self.attn = MixedAttention(
            dim,
            num_heads=num_heads,
            num_vanilla_heads=num_vanilla_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            **dd
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            **dd,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformer(nn.Module):
    """Fixed-grid DeiT-III encoder with a CLS classifier."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, num_vanilla_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_norm=True, init_values=1e-6,
                 drop_rate=0., pos_drop_rate=0., proj_drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., device=None, dtype=None):
        super().__init__()
        if not 0 <= num_vanilla_heads <= num_heads:
            raise ValueError('num_vanilla_heads must be between zero and num_heads')
        dd = dict(device=device, dtype=dtype)
        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1
        self.global_pool = 'token'
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, **dd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, **dd))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim, **dd) * .02)
        self.pos_drop = nn.Dropout(pos_drop_rate)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, num_vanilla_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_norm=qk_norm, init_values=init_values,
                  proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], **dd)
            for i in range(depth)
        ])
        self.norm = LayerNorm(embed_dim, **dd)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        # Depth-first traversal matches timm's named_apply initialization order.
        def initialize(module):
            for child in module.children():
                initialize(child)
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        initialize(self)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return self.norm(self.blocks(x))

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(self.fc_norm(x[:, 0]))
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))
