import torch
import torch.nn as nn

class VisualAttention(nn.Module):
    def __init__(self, D_in, D_model):
        super().__init__()
        self.W_q = nn.Linear(D_in, D_model)
        self.W_k = nn.Linear(D_in, D_model)
        self.W_v = nn.Linear(D_in, D_model)
        self.query_projector = nn.Sequential(
            nn.Linear(D_in, D_in),
            nn.GELU(),
            nn.Linear(D_in, D_in)
        )
    def forward(self, query_input, visual_bank, return_attn_weights=True):
        query_input = self.query_projector(query_input)  # [B, T_q, D_in]
        B, T_q, _ = query_input.shape
        N, T_k, _ = visual_bank.shape

        # Project to Q, K, V
        Q = self.W_q(query_input)          # [B, T_q, D_model]
        visual_bank = visual_bank.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, T_k, D_in]
        visual_bank = visual_bank.reshape(B, N * T_k, -1)  # [B, N*T_k, D_in]

        K = self.W_k(visual_bank)          # [B, N*T_k, D_model]
        V = self.W_v(visual_bank)          # [B, N*T_k, D_model]

        # Attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, T_q, N*T_k]
        attn_scores = attn_scores / (Q.shape[-1] ** 0.5)  # scale

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T_q, N*T_k]

        # Weighted sum
        context = torch.bmm(attn_weights, V)  # [B, T_q, D_model]
        if return_attn_weights:
            importance = attn_weights.mean(dim=(0, 1))  # shape: [N*T_k]
            importance = importance.view(N, T_k)
            importance = importance.mean(dim=1)
            sorted_indices = torch.argsort(importance, descending=True)  # [N]
            important_batch_indices = sorted_indices.tolist()
            return context, important_batch_indices
        return context
