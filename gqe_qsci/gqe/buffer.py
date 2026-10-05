# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Modifications Copyright (c) 2026 Ryota Kemmoku
# Modified from the original file in NVIDIA CUDA-QX.
# Changes made: store `log_prob` and the optional diffusion trajectory
# (reveal_step) in the replay buffer.


from collections import deque
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class ReplayBuffer:
    def __init__(self, size=sys.maxsize, capacity=1000000):
        self.size = size
        self.buf = deque(maxlen=capacity)

    # CLAUDE
    def push(self, seq, energy, old_log_probs, reveal_step=None):
        """
        Store one rollout sample.

        reveal_step : (L,) long tensor — the absorbing-diffusion trajectory,
                i.e. the timestep at which each gate position was committed
                (state["reveal_step"] from sample_sequence). log_prob() needs
                it to score the exact path that was sampled.
                None for models whose trajectory is implicit in the gate
                sequence itself (GPT-2, single-shot, DAG GNN).
        """
        self.buf.append((seq, energy, old_log_probs, reveal_step))
        if len(self.buf) > self.size:
            self.buf.popleft()
            
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)
            
    def load(self, path):
        with open(path, "rb") as f:
            self.buf = pickle.load(f)
            
    # CLAUDE
    def __getitem__(self, idx):
        item = self.buf[idx]
        seq, energy, old_log_probs, reveal_step = item
        return {
            "idx": seq,
            "energy": energy,
            "old_log_probs": old_log_probs,
            "reveal_step": reveal_step,   # None for non-absorbing models
        }

    def __len__(self):
        return len(self.buf)


class BufferDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer, repetition):
        self.buffer = buffer
        self.repetition = repetition

    def __getitem__(self, idx):
        idx = idx % len(self.buffer)
        sample = self.buffer[idx]
        return {
            "idx": sample["idx"],
            "energy": sample["energy"],
            "old_log_probs": sample["old_log_probs"],
            "reveal_step": sample["reveal_step"],   # None for non-absorbing models
        }
    
    # CLAUDE
    def __len__(self):
        return len(self.buffer) * self.repetition


# CLAUDE
def buffer_collate_fn(batch):
    """
    Custom collate function for BufferDataset.

    Handles the optional 'reveal_step' field: an (L,) long tensor for the
    absorbing-diffusion models and None for all others (GPT-2, single-shot,
    DAG GNN). PyTorch's default collate cannot mix tensors and None values, so
    we strip it out, collate the rest normally, then re-attach.

    If all are None   → collated["reveal_step"] = None
    If all are tensors → collated["reveal_step"] = stacked (B, L) tensor
    Mixed is a bug (a buffer holding rollouts from more than one model type)
    and raises instead of being papered over with zero-filled trajectories,
    which would silently score those samples against a wrong path.
    """
    rs_list = [item["reveal_step"] for item in batch]
    rest = [{k: v for k, v in item.items() if k != "reveal_step"} for item in batch]
    collated = default_collate(rest)

    n_none = sum(r is None for r in rs_list)
    if n_none not in (0, len(rs_list)):
        # Deliberately a raise, not an assert: this guards correctness, and
        # asserts are stripped under `python -O`.
        raise ValueError(
            f"buffer_collate_fn: mixed reveal_step in batch "
            f"({n_none}/{len(rs_list)} are None). A replay buffer must hold "
            f"rollouts from a single model type - absorbing-diffusion samples "
            f"carry a trajectory, GPT-2 / single-shot / DAG GNN samples do not."
        )

    if n_none:
        collated["reveal_step"] = None
    else:
        collated["reveal_step"] = torch.stack(rs_list, dim=0)

    return collated
