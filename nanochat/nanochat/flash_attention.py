"""Flash Attention 3 with a PyTorch SDPA fallback."""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface
            else:
                return None

    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

from nanochat.common import COMPUTE_DTYPE
USE_FA3 = HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16


def causal_attention(q, k, v, window_size):
    """Causal attention on (batch, sequence, heads, channels), with a left window."""
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    length = q.size(2)
    window = window_size[0]
    if window < 0 or window >= length:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        row = torch.arange(length, device=q.device).unsqueeze(1)
        col = torch.arange(length, device=q.device).unsqueeze(0)
        mask = (col <= row) & ((row - col) <= window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return y.transpose(1, 2)
