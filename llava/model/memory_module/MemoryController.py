import math
from typing import Optional, List, Tuple
import torch
from torch import nn
from transformers.activations import ACT2FN

class Config:
    mm_hidden_size = 896
    mm_hidden_act = "relu"
    mm_num_attention_heads = 8
    patch_size = 196
    mm_attention_probs_dropout_prob = 0.1
    mm_layer_norm_eps = 1e-12
    mm_hidden_dropout_prob = 0.1
    mm_intermediate_size = 4 * mm_hidden_size
    num_memory_tokens = 8
    depth = 1
    mm_dtype = torch.float16

class Residual(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size, dtype=config.mm_dtype)
        self.layernorm = nn.LayerNorm(output_size, eps=config.mm_layer_norm_eps, dtype=config.mm_dtype)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.mm_hidden_size
        self.num_attention_heads = config.mm_num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.residual = Residual(self.hidden_size, self.hidden_size, config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, kv_hidden_states=None, output_attentions=True):
        query = self.transpose_for_scores(self.q_proj(hidden_states))
        key = self.transpose_for_scores(self.k_proj(kv_hidden_states if kv_hidden_states is not None else hidden_states))
        value = self.transpose_for_scores(self.v_proj(kv_hidden_states if kv_hidden_states is not None else hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = torch.nn.functional.softmax(scores, dim=-1)  #(batch_size, num_heads, query_len, key_len)
        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.size(0), -1, self.hidden_size)
        output = self.residual(context, hidden_states)
        return output, probs

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_segment_fusion_attention = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_intermediate_size, dtype=config.mm_dtype),
            ACT2FN[config.mm_hidden_act]
        )
        self.residual = Residual(config.mm_intermediate_size, config.mm_hidden_size, config)

    def forward(self, query_states, kv_states):
        attention_output, attention_probs = self.memory_segment_fusion_attention(query_states, kv_hidden_states=kv_states, output_attentions=True)
        layer_output = self.residual(self.mlp(attention_output), attention_output)
        return layer_output, attention_probs

class TransformerProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        self.layers = nn.ModuleList([TransformerLayer(self.config) for _ in range(self.config.depth)])
        self.num_memory_tokens = self.config.num_memory_tokens
        self.hidden_size = self.config.mm_hidden_size
        self.patch_size = self.config.patch_size
        self.initial_memory = nn.Parameter(torch.empty(self.num_memory_tokens, self.patch_size, self.hidden_size))
        nn.init.xavier_uniform_(self.initial_memory)
        self.memory_cache: List[torch.Tensor] = []
        self.memory_update_attention = Attention(self.config)
        self.frame_attn_scores: List[torch.Tensor] = []

    def _update_memory_tokens_with_cache(self, current_memory: torch.Tensor) -> torch.Tensor:
        if not self.memory_cache:
            return current_memory
        past_memory = torch.cat(self.memory_cache, dim=0).unsqueeze(0)
        query = current_memory.unsqueeze(0)
        B, Lq, P, D = query.shape
        query_2d = query.view(B, Lq * P, D)
        keyval_2d = past_memory.view(1, -1, D)
        updated_2d, memory_evolution_prob = self.memory_update_attention(query_2d, kv_hidden_states=keyval_2d)
        print(f"memory_evolution_prob shape: {memory_evolution_prob.shape}")
        probs_sum = memory_evolution_prob.sum(dim=1).sum(dim=1).squeeze(0)  # [Lq * P]
        print(f"probs_sum shape: {probs_sum.shape}")
        # Now: sum attention to each chunk
        attn_per_chunk = probs_sum.split(196, dim=-1)  # list of N tensors (B, S_q, 196)

        # For each chunk, sum over the key_len axis (last dim)
        attn_chunk_sums = [chunk.sum(dim=-1) for chunk in attn_per_chunk]  # list of (B, S_q)

        # Stack to get (B, S_q, N)
        attn_chunk_map = torch.stack(attn_chunk_sums, dim=-1)  # shape: (B, S_q, N)
        print(f"attn_chunk_map shape: {attn_chunk_map.shape}")
        print(f"attn_chunk_map: {attn_chunk_map}")
        updated_4d = updated_2d.view(B, Lq, P, D)
        return updated_4d.squeeze(0)


    def forward(self, image_features: torch.Tensor):
        device = image_features.device
        dtype = image_features.dtype
        B = 1
        F, P, D = image_features.shape
        memory_tokens = self.initial_memory.to(device=device, dtype=dtype)
        if self.memory_cache:
            memory_tokens = self.memory_cache[-1]

        if len(self.memory_cache) > 1:
            memory_tokens = self._update_memory_tokens_with_cache(memory_tokens)
            print("memory update called")
        memory_2d = memory_tokens.reshape(B, self.num_memory_tokens * P, D)
        image_2d = image_features.reshape(B, F * P, D)
        frame_attn_scores = []

        for layer in self.layers:
            memory_2d, attn_probs = layer(memory_2d, image_2d)
            # print(f"attn_probs shape: {attn_probs.shape}")
            attn_sum = attn_probs.sum(dim=1).sum(dim=1).squeeze(0)  # [F * P]
            # print(f"attn_sum shape: {attn_sum.shape}", "attn_sum:", attn_sum[:32])
            frame_scores = attn_sum.view(F, P).mean(dim=1)
            # print(f"frame_scores shape: {frame_scores.shape}")
            frame_attn_scores.append(frame_scores)

        final_memory = memory_2d.view(B, self.num_memory_tokens, P, D)
        self.memory_cache.append(final_memory.squeeze(0))
        if len(self.memory_cache) > 10:
            self.memory_cache = self.memory_cache[-10:]

        final_score = frame_attn_scores[-1]
        self.frame_attn_scores.append(final_score.detach())
        return self.memory_cache, self.frame_attn_scores


