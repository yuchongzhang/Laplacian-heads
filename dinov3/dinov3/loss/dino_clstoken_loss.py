# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from dinov3.distributed import get_process_subgroup, get_subgroup_size


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def init_weights(self) -> None:
        self.center.zero_()


    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        world_size = get_subgroup_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops, batch, prototypes]
        teacher_probs:  [teacher crops, batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)
