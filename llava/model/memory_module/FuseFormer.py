import torch.nn as nn
import torch
class FuseFormer(nn.Module):
    """
    A tiny transformer that lets `memory` read the question before
    either of them see the image.  All Qwen & vision weights stay frozen.
    """
    def __init__(self, dim=896, heads=8, layers=2, ffn_mult=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=ffn_mult*dim,
                batch_first=True, norm_first=True
            ) for _ in range(layers)
        ])
        # optional trainable scale/shift to match statistics
        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False)
        )

    def forward(self, query_emb, memory_emb):
        # 1) project query into the same stats space as memory
        q = self.query_proj(query_emb)             # [1, q, 896]
        # 2) concat and run a few self-attn layers
        frame, patch, dim = memory_emb.shape
        memory_emb = memory_emb.reshape(1, -1, dim)  # [1, M, 896]
        x = torch.cat([q, memory_emb], dim=1)      # [B, q+M, 896]
        for blk in self.blocks:
            x = blk(x)
        q_out, mem_out = x.split([q.size(1), memory_emb.size(1)], dim=1)
        mem_out = mem_out.reshape(frame, patch, dim)
        return q_out, mem_out
