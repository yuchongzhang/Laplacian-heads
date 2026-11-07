"""Select fused scaled-dot-product attention."""
import os
import torch
_fused_attn = int(os.environ.get('TIMM_FUSED_ATTN', '1'))
def use_fused_attn():
    return bool(_fused_attn) and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
def set_fused_attn(enable=True):
    global _fused_attn
    _fused_attn = int(enable)
