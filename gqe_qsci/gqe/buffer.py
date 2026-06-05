# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Modifications Copyright (c) 2026 Ryota Kemmoku
# Modified from the original file in NVIDIA CUDA-QX.
# Changes made: store `log_prob` and optional diffusion masks in the replay buffer.


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

    def push(self, seq, energy, old_log_probs, masks=None):
        """
        Store one rollout sample.

        masks : (B, T, L) bool tensor — pre-sampled diffusion corruption masks
                returned by CircuitDiffusionModelAbsorbing.sample_masks().
                None for non-diffusion models (GPT-2, SimpleGNN, …).
        """
        self.buf.append((seq, energy, old_log_probs, masks))
        if len(self.buf) > self.size:
            self.buf.popleft()
            
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)
            
    def load(self, path):
        with open(path, "rb") as f:
            self.buf = pickle.load(f)
            
    def __getitem__(self, idx):
        item = self.buf[idx]
        seq, energy, old_log_probs, masks = item
        return {
            "idx": seq,
            "energy": energy,
            "old_log_probs": old_log_probs,
            "masks": masks,   # may be None for non-absorbing models
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
            "masks": sample["masks"],   # may be None for non-absorbing models
        }
    
    def __len__(self):
        return len(self.buffer) * self.repetition


def buffer_collate_fn(batch):
    """
    Custom collate function for BufferDataset.

    Handles the optional 'masks' field which is a (T, L) bool tensor for
    absorbing-diffusion models and None for all other models (GPT-2, Simple).
    PyTorch's default collate cannot mix tensors and None values, so we
    strip out masks, collate the rest normally, then re-attach them.

    If all masks are None  → collated["masks"] = None
    If all masks are tensors → collated["masks"] = stacked (B, T, L) tensor
    Mixed (shouldn't happen in practice) → None entries filled with zeros.
    """
    masks_list = [item["masks"] for item in batch]
    non_mask_batch = [{k: v for k, v in item.items() if k != "masks"} for item in batch]
    collated = default_collate(non_mask_batch)

    if all(m is None for m in masks_list):
        collated["masks"] = None
    else:
        ref = next(m for m in masks_list if m is not None)
        filled = [
            m if m is not None else torch.zeros_like(ref)
            for m in masks_list
        ]
        collated["masks"] = torch.stack(filled, dim=0)

    return collated