"""Observe the exact tensor used by every language measurement in the paper."""

import torch


class PenultimateMLPInput:
    """Capture the RMS-normalized post-attention residual in block N-2.

    Block.forward computes x = x + attention(norm(x)), then mlp(norm(x)).
    Its MLP input is exactly the RMS-normalized residual stream after attention
    in the second-to-last block, with the native dtype and RMSNorm epsilon.
    """

    def __init__(self, model):
        if model.config.n_layer < 2:
            raise ValueError("At least two blocks are needed")
        self.model = model
        self.value = None
        self.handle = model.transformer.h[-2].mlp.register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        self.value = inputs[0].detach()

    @torch.inference_mode()
    def __call__(self, inputs):
        self.value = None
        self.model.forward_features(inputs)
        if self.value is None:
            raise RuntimeError("Penultimate MLP hook was not called")
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.handle.remove()
        self.value = None
