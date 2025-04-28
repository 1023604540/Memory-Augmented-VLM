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
        )

        # Optional sinusoidal temporal positional encoding
        self.use_pe = use_positional_encoding
        if use_positional_encoding:
            # register as buffer so it moves with the module but is not trainable
            self.register_buffer(
                'temporal_pe',
                build_sine_time_table(max_frames, input_dim, device=torch.device('cpu'))
            )

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: Tensor of shape [F, P, D]
                          F = number of frames
                          P = number of patch tokens per frame (e.g. 729)
                          D = feature dimension (e.g. 1152)
        Returns:
            enriched_feats: Tensor of shape [F, P, D], where each patch token
                            has had the GRU’s bidirectional temporal context added.
        """
        F, P, D = visual_feats.shape
        device = visual_feats.device

        # 1. Pool patches to get one vector per frame: [F, D]
        frame_vecs = visual_feats.mean(dim=1)

        # 2. Add sinusoidal temporal encoding if enabled
        if self.use_pe:
            pe = self.temporal_pe[:F].to(device)  # [F, D]
            frame_vecs = frame_vecs + pe

        # 3. Prepare sequence for GRU: [T=F, B=1, D]
        seq = frame_vecs.unsqueeze(1)

        # 4. Run through GRU
        output, _ = self.gru(seq)  # output: [F, 1, 2*hidden_size] == [F, 1, D]
        print(f"BiGRU output shape: {output.shape}")
        # 5. Remove batch dim and broadcast back to patches
        frame_ctx = output.squeeze(1)            # [F, D]
        temporal_term = frame_ctx.unsqueeze(1)   # [F, 1, D]
        temporal_term = temporal_term.expand(-1, P, -1)  # [F, P, D]

        # 6. Add temporal context onto original visual features
        enriched_feats = visual_feats + temporal_term     # [F, P, D]
        print(f"BiGRU enriched features shape: {enriched_feats.shape}")
        return enriched_feats
