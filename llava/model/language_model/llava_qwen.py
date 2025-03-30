#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
from transformers.cache_utils import Cache, DynamicCache

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        print("before past_key_values")
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                print(f"Layer {layer_idx}: key shape = {key.shape}, value shape = {value.shape}")
        print("after past_key_values")

        if past_key_values is not None:
            if self.model.memory_readout_cache is not None:
                memory_readout = self.model.memory_readout_cache.to(dtype=self.dtype, device=self.device)
                T_mem = memory_readout.shape[0]  # memory tokens
                B = input_ids.shape[0]

                # # === 1. Expand attention mask ===
                # if attention_mask is not None:
                #     memory_mask = torch.ones(B, T_mem, dtype=attention_mask.dtype, device=attention_mask.device)
                #     attention_mask = torch.cat([memory_mask, attention_mask], dim=1)
                #     inputs["attention_mask"] = attention_mask

                # === 3. Inject past_key_values ===
                past_key_values = self.inject_memory_as_kv(memory_readout, past_key_values)
                # inputs["past_key_values"] = past_key_values

                self.model.memory_readout_cache = None
        print("before past_key_values")
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                print(f"Layer {layer_idx}: key shape = {key.shape}, value shape = {value.shape}")
        print("after past_key_values")
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        print(f"inputs coming, {input_ids.shape}")
        print("position_ids coming", {inputs["position_ids"]})
        # old_cache = inputs.get("past_key_values", None)

        # inputs["position_ids"] = None
        # inputs["cache_position"] = None
        #Inject memory into past_key_values
        # if self.model.memory_readout_cache is not None:
        #     memory_readout = self.model.memory_readout_cache.to(dtype=self.dtype, device=self.device)
        #     T_mem = memory_readout.shape[0]  # memory tokens
        #     B = input_ids.shape[0]
        #
        #     # === 1. Expand attention mask ===
        #     if attention_mask is not None:
        #         memory_mask = torch.ones(B, T_mem, dtype=attention_mask.dtype, device=attention_mask.device)
        #         attention_mask = torch.cat([memory_mask, attention_mask], dim=1)
        #         inputs["attention_mask"] = attention_mask
        #     # inputs["position_ids"] = None
        #     # inputs["cache_position"] = None
        #     #
        #     # # === 2. Expand position_ids ===
        #     # if position_ids is None:
        #     #     start_pos = T_mem
        #     #     memory_pos = torch.arange(start_pos, start_pos + input_ids.shape[1], dtype=self.dtype,
        #     #                               device=self.device)
        #     #     print("memory_pos:", memory_pos.shape)
        #     #     memory_pos = memory_pos.unsqueeze(0).expand(B, -1)
        #     #     inputs["position_ids"] = memory_pos
        #
        #     # === 3. Inject past_key_values ===
        #     past_key_values = self.inject_memory_as_kv(memory_readout)
        #     inputs["past_key_values"] = past_key_values
        #
        #     # === 4. Manually update cache position ===
        #     # Qwen2 supports `cache_position` kwarg to align KV cache
        #     # inputs["cache_position"] = torch.arange(T_mem, T_mem + input_ids.shape[1],
        #     #                                         device=input_ids.device).unsqueeze(0)
        #     # # for i, (k, v) in enumerate(past_key_values):
        #     #     print(f"Layer {i}: key shape {k.shape}, value shape {v.shape}")
        #     # print("Expanded attention mask:", inputs["attention_mask"].shape)
        #     # print("cache_position:", inputs.get("cache_position", None))
        #     # ✅ Clear cache
        #     self.model.memory_readout_cache = None

        return inputs

    #
    # def inject_memory_as_kv(self, memory_readout, ):
    #     B = 1
    #     D = memory_readout.size(-1)
    #     H = self.config.num_key_value_heads  # number of attention heads is 14, kv heads is 2
    #     L = self.config.num_hidden_layers  # number of Transformer layers
    #     Dh = 64
    #     T = memory_readout.shape[0]  # number of memory tokens
    #
    #     cache = DynamicCache()
    #
    #     for i in range(L):
    #         key = self.model.memory_key_projs[i](memory_readout).view(B, T, H, Dh)
    #         key = key.permute(0, 2, 1, 3).contiguous()
    #         value = self.model.memory_value_projs[i](memory_readout).view(B, T, H, Dh)
    #         value = value.permute(0, 2, 1, 3).contiguous()
    #         # print("memory_readout:", memory_readout.shape)
    #         # print("key shape", key.shape)
    #         cache.update(
    #             key_states=key,
    #             value_states=value,
    #             layer_idx=i
    #         )
    #
    #
    #     return cache
    def inject_memory_as_kv(self, memory_readout, old_cache=None):
        """
        Example function to manually concatenate memory K/V to an existing
        list-based cache of shape [B, n_heads, seq_len, head_dim].

        If old_cache is None or empty, we treat it as zero-length for each layer.
        """
        B = 1  # your batch size, or read from memory_readout if needed
        T = memory_readout.shape[0]  # number of memory tokens
        H = self.config.num_key_value_heads
        L = self.config.num_hidden_layers
        Dh = 64  # typical dimension per head for Qwen; confirm from your config

        new_cache = []

        # We'll do one pass over the L layers.
        for i in range(L):
            old_key, old_value = old_cache[i]

            # If there's no old key/value, define them as zero-length on dim=2
            if old_key is None or old_value is None:
                old_key = torch.empty(
                    B, H, 0, Dh, dtype=memory_readout.dtype, device=memory_readout.device
                )
                old_value = torch.empty_like(old_key)  # same shape/dtype/device

            # 1) Compute memory key/value for this layer
            #    from your memory projections. Right now, they produce [B, T, H, Dh],
            #    so we permute them to [B, H, T, Dh].
            mem_key = self.model.memory_key_projs[i](memory_readout).view(B, T, H, Dh)
            mem_key = mem_key.permute(0, 2, 1, 3).contiguous()  # [B, H, T, Dh]
            mem_value = self.model.memory_value_projs[i](memory_readout).view(B, T, H, Dh)
            mem_value = mem_value.permute(0, 2, 1, 3).contiguous()

            # 2) Concatenate memory + old. Decide if memory tokens go before or after the old tokens.
            #    Typically for "prepend" (memory is first):
            new_key = torch.cat([mem_key, old_key], dim=2)  # shape [B, H, T+old_len, Dh]
            new_value = torch.cat([mem_value, old_value], dim=2)

            # 3) Append the new (key, value) to new_cache
            new_cache.append((new_key, new_value))

        return new_cache


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
