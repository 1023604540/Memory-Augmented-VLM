import torch.nn as nn


class MemoryFuser(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, device="cuda"):
        super(MemoryFuser, self).__init__()
        self.device = device

        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, memory_tokens):
        """
        memory_tokens: Tensor of shape (batch_size, num_segments, hidden_dim)
        """
        x = self.input_proj(memory_tokens)
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        return x
