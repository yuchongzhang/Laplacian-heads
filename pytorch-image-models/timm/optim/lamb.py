"""LAMB optimizer. Modifications Copyright 2021 Ross Wightman."""
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer



class Lamb(Optimizer):
    """LAMB with bias correction, gradient averaging, and global norm clipping."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01):
        defaults = dict(lr=lr, bias_correction=True, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=True, max_grad_norm=1.0, trust_clip=False, always_adapt=False, caution=False, decoupled_decay=False, corrected_weight_decay=False)
        super().__init__(params, defaults)

    def _get_clip_grad_norm(self):
        max_grad_norm = self.defaults['max_grad_norm']
        if max_grad_norm is None:
            return None
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')
                norms.append(torch.linalg.vector_norm(grad))
        global_norm = torch.linalg.vector_norm(torch.stack(norms))
        clip_global_norm = (global_norm / max_grad_norm).clamp_(min=1.0)
        return clip_global_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        clip_grad_norm = self._get_clip_grad_norm()
        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            if bias_correction:
                bias_correction1 = 1 - beta1 ** group['step']
                bias_correction2 = 1 - beta2 ** group['step']
            else:
                bias_correction1, bias_correction2 = (1.0, 1.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if clip_grad_norm is not None:
                    grad.div_(clip_grad_norm)
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)
                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)
                if weight_decay != 0:
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    trust_ratio = w_norm / g_norm
                    trust_ratio = torch.where(w_norm > 0, torch.where(g_norm > 0, trust_ratio, 1.0), 1.0)
                    update.mul_(trust_ratio)
                p.add_(update, alpha=-group['lr'])
        return loss
