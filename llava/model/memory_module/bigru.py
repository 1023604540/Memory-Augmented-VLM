import math
import torch
import torch.nn as nn

def build_sine_time_table(max_frames: int, dim: int, device: torch.device):
    pe = torch.zeros(max_frames, dim, device=device)
    pos = torch.arange(max_frames, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device) *
                    (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

class TemporalGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 896,
        hidden_size: int = 448,      # so 2*448 == 896
        num_layers: int = 1,
        bidirectional: bool = True,
        max_frames: int = 300,
        use_positional_encoding: bool = False
    ):
        super().__init__()
        self.use_pe = use_positional_encoding

        # 1-layer Bi-GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False,
        )

        if self.use_pe:
            # non-trainable sinusoidal PE
            self.register_buffer(
                "temporal_pe",
                build_sine_time_table(max_frames, input_dim, device=torch.device("cpu")),
            )

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        # visual_feats: [F, P, D]
        F, P, D = visual_feats.shape

        # 1. Mean-pool across patches → [F, D]
        frame_vecs = visual_feats.mean(dim=1)

        # 2. Optional positional encoding
        if self.use_pe:
            pe = self.temporal_pe[:F].to(frame_vecs.device)
            frame_vecs = frame_vecs + pe

        # 3. Format for GRU input and cast to float32
        seq = frame_vecs.unsqueeze(1).to(torch.float32)  # ← critical fix

        # 4. GRU forward
        output, _ = self.gru(seq)  # float32 input guaranteed

        # 5. Broadcast GRU output back to [F, P, D]
        temporal_term = output.squeeze(1).unsqueeze(1).expand(-1, P, -1)
        temporal_term = temporal_term.to(visual_feats.dtype)

        # 6. Residual add
        enriched_feats = visual_feats + temporal_term
        return enriched_feats

