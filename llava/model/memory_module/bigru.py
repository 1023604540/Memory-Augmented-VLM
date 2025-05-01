import math
import torch
import torch.nn as nn

def build_sine_time_table(max_frames: int, dim: int, device: torch.device):
    """
    Create a sinusoidal positional encoding table for temporal positions.
    Returns a tensor of shape [max_frames, dim].
    """
    pe = torch.zeros(max_frames, dim, device=device)
    pos = torch.arange(max_frames, dtype=torch.float32, device=device).unsqueeze(1)  # [max_frames, 1]
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # [max_frames, dim]

class TemporalGRUEncoder(nn.Module):
    """
    Takes SigLIP visual features of shape [F, P, D] (frames × patches × dim)
    and returns enriched features of the same shape, where each patch
    token has added temporal context from a 1-layer Bi-GRU.
    """
    def __init__(
        self,
        input_dim: int = 1152,
        hidden_size: int = 576,
        num_layers: int = 1,
        bidirectional: bool = True,
        max_frames: int = 300,
        use_positional_encoding: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 1-layer bidirectional GRU where 2*hidden_size == input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False,
        ).float()

        # Optional sinusoidal temporal positional encoding
        self.use_pe = use_positional_encoding
        if use_positional_encoding:
            # register as buffer so it moves with the module but is not trainable
            self.register_buffer(
                'temporal_pe',
                build_sine_time_table(max_frames, input_dim, device=torch.device('cpu'))
            )

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        # visual_feats is likely bfloat16 under autocast…
        F, P, D = visual_feats.shape

        # 1) pool patches to [F, D]
        frame_vecs = visual_feats.mean(dim=1)

        # 2) add PE if you want…
        if self.use_pe:
            pe = self.temporal_pe[:F].to(frame_vecs.device)
            frame_vecs = frame_vecs + pe

        # 3) shape for GRU: [T, B, D]
        seq = frame_vecs.unsqueeze(1)   # [F, 1, D]

        # ───── FIX ─────
        # cast up to float32 so the fused kernel is not selected
        dtype_in = seq.dtype
        seq = seq.to(torch.float32)

        # 4) run GRU in float32
        output, _ = self.gru(seq)       # output: [F, 1, 2*hidden]

        # cast back to original dtype (bfloat16) if you like
        output = output.to(dtype_in)
        # ──────────────────

        # 5) broadcast back to patches
        frame_ctx = output.squeeze(1)           # [F, D]
        temporal_term = frame_ctx.unsqueeze(1)      # [F, 1, D]
        temporal_term = temporal_term.expand(-1, P, -1)

        enriched_feats = visual_feats + temporal_term  # [F, P, D]
        return enriched_feats
