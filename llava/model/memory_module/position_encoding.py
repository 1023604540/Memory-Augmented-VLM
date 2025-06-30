"""
temporal_positional_encoding.py

Script to add temporal positional encoding to visual feature vectors.

Visual feature tensor shape: (T, N, C) or (B, T, N, C).
"""

import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, max_frames, embed_dim, learnable=True):
        """
        Args:
            max_frames (int): max possible video length (should be >= longest input video)
            embed_dim (int): feature dimension
            learnable (bool): use learnable embedding if True, else use fixed sinusoidal
        """
        super().__init__()
        self.max_frames = max_frames
        self.embed_dim = embed_dim
        self.learnable = learnable

        if learnable:
            self.frame_embed = nn.Embedding(max_frames, embed_dim)
        else:
            pe = torch.zeros(max_frames, embed_dim, dtype=torch.float32)
            position = torch.arange(0, max_frames).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('frame_embed', pe)

    def forward(self, x, frame_indices=None):
        """
        Args:
            x (Tensor): shape (T, N, C) or (B, T, N, C)
            frame_indices (Tensor or None): shape (T,) or (B, T), values in [0, original_video_length)
        Returns:
            Tensor with positional encoding added.
        """
        if frame_indices is None:
            # Default to using 0...T-1
            if x.dim() == 3:
                T = x.size(0)
                frame_indices = torch.arange(T, device=x.device)
            elif x.dim() == 4:
                B, T = x.size(0), x.size(1)
                frame_indices = torch.arange(T, device=x.device).expand(B, T)
            else:
                raise ValueError(f'Expected 3D or 4D input, got {x.dim()}D.')

        if x.dim() == 3:
            pe = self._get_pe(frame_indices, x.device).to(x.dtype)  # (T, C)
            if torch.isnan(pe).any():
                raise ValueError("Positional encoding contains NaN values.")
            # x_norm = x.norm(dim=-1).mean().item()
            # pe_norm = pe.norm(dim=-1).mean().item()
            # print(f"Check Magnitude: Feature norm: {x_norm:.3f}, PE norm: {pe_norm:.3f}")
            return x + pe[:, None, :]  # (T, N, C)
        elif x.dim() == 4:
            pe = self._get_pe(frame_indices, x.device)  # (B, T, C)
            return x + pe[:, :, None, :]  # (B, T, N, C)
        else:
            raise ValueError(f'Expected 3D or 4D input, got {x.dim()}D.')

    def _get_pe(self, indices, device):
        indices = indices.to(device)
        if torch.any(indices >= self.max_frames):
            raise ValueError(f"indices exceed max_frames: max {indices.max().item()} vs limit {self.max_frames}")
        if torch.any(indices < 0):
            raise ValueError(f"indices contains negative values: min {indices.min().item()}")
        if self.learnable:
            return self.frame_embed(indices)
        else:
            return self.frame_embed[indices]
