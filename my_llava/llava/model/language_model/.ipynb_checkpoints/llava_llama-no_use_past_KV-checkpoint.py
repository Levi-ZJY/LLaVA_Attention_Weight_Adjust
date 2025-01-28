#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# from ..Custom_Llama import CustomLlamaModel

# -------------------------------------------------------------------------------------------------------------------

import inspect

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaSdpaAttention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import math
import copy
import numpy as np

# 自定义 LlamaAttention (继承原版)
class My_LlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # print("---Using CustomLlamaAttention---")
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        Now_img_txt_token_spans = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # outputs = super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        # return outputs
        """!!!"""
        # print("My_LlamaAttention---Now_img_txt_token_spans:", Now_img_txt_token_spans)   # 成功！！！
        assert len(Now_img_txt_token_spans)==1, "Error from My_LlamaAttention"
        """!!!"""

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)



        """!!!"""

        
        # pyorch,python中等于号是 别名 不是拷贝！且函数传参也是别名！用 = 则会修改所有上层的Now_img_txt_token_spans
        All_Now_img_txt_token_spans = copy.deepcopy(Now_img_txt_token_spans) 
        
        last_term = All_Now_img_txt_token_spans[0][-1]
        # print("last_term:", last_term) 

        now_len_alltokens = attn_weights.size()[-1]
        
        if last_term[0]=="text":
            last_term[-1] = now_len_alltokens
            All_Now_img_txt_token_spans[0][-1] = last_term
            
        elif last_term[0]=="image":
            print("---Check this!!!---")
            new_term = ('text', last_term[-1], now_len_alltokens)
            All_Now_img_txt_token_spans[0].append(new_term)
        else:
            print("---Error!!!---")
        
        
        
        # print("Change---All_Now_img_txt_token_spans:", All_Now_img_txt_token_spans)
        print("attn_weights.size():", attn_weights.size())   # torch.Size([1, num_heads, 1, num_tokens_now])  -> torch.Size([1, 40, 1, num_tokens_now])


        # print(attn_weights)

        
        """!!!"""

        
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




class My_LlamaAttention_SDPA(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # print("---Using CustomLlamaAttention---")
        super().__init__(config, layer_idx)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        Now_img_txt_token_spans = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        """!!!"""
        # print("My_LlamaAttention_SDPA---Now_img_txt_token_spans:", Now_img_txt_token_spans)   # 成功！！！
        assert len(Now_img_txt_token_spans)==1, "Error from My_LlamaAttention_SDPA"
        """!!!"""
        
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # print("query_states:", query_states.size())
        # print("key_states:", key_states.size())
        # print("value_states:", value_states.size())

        # print("Q_K_V_states:", query_states.size(), key_states.size(), value_states.size())
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # print("kv_seq_len:", kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()


        """!!!"""

        
        # print("query_states", query_states.size())
        # print("key_states", key_states.size())
        # print("value_states", value_states.size())


        # pyorch,python中等于号是 别名 不是拷贝！且函数传参也是别名！用 = 则会修改所有上层的Now_img_txt_token_spans
        All_Now_img_txt_token_spans = copy.deepcopy(Now_img_txt_token_spans) 
        
        last_term = All_Now_img_txt_token_spans[0][-1]
        # print("last_term:", last_term) 

        now_len_alltokens = key_states.size()[-2]
        
        if last_term[0]=="text":
            last_term[-1] = now_len_alltokens
            All_Now_img_txt_token_spans[0][-1] = last_term
            
        elif last_term[0]=="image":
            print("---Check this!!!---In My_LlamaAttention_SDPA")
            new_term = ('text', last_term[-1], now_len_alltokens)
            All_Now_img_txt_token_spans[0].append(new_term)
        else:
            print("---Error!!!---In My_LlamaAttention_SDPA")
        
        
        # print("Change---All_Now_img_txt_token_spans:", All_Now_img_txt_token_spans)


        
        """!!!"""

        """
        print("attention_mask", attention_mask)  # attention_mask恒为 None
        # print(attention_mask.size())

        print("self.is_causal", self.is_causal)   # self.is_causal恒为 True
        print("q_len", q_len)

        # 这里在生成第一个token时，因为q_len=638，所以final_is_causal为True。之后因为q_len都为 1，所以final_is_causal为 False
        t_final_is_causal = self.is_causal and attention_mask is None and q_len > 1 
        print("!!!final_is_causal!!!", t_final_is_causal)
        """

        
        """!!!"""

        # TBD 此处或在之前添加设置attention_mask，使其屏蔽image tokens

        """!!!"""


        # print("attention_mask_4", attention_mask)                                 # 这里attention_mask恒等于 None
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )


        """!!!"""
        # print("attn_output", attn_output.size())
        # print()
        """!!!"""
        

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


        
# 自定义 LlamaDecoderLayer (继承原版)
class My_LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        """
        print("-----")
        print(config._attn_implementation)
        config._attn_implementation = "eager"   # 注意！！！强制指定attention计算为"eager"模式
        print(config._attn_implementation)
        print("-----")
        """
        
        # print("---Using CustomLlamaDecoderLayer---")
        super().__init__(config, layer_idx)

        print("---!!!Using My_LlamaAttention_SDPA!!!---")
        self.self_attn = My_LlamaAttention_SDPA(config=config, layer_idx=layer_idx)
        
        # self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        # self.self_attn = My_LlamaAttention(config=config, layer_idx=layer_idx)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        Now_img_txt_token_spans = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # outputs = super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        # return outputs

        """!!!"""
        # print("My_LlamaDecoderLayer---Now_img_txt_token_spans:", Now_img_txt_token_spans)
        """!!!"""

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        """!!!"""

        # print("attention_mask_3", attention_mask)                    # 这里attention_mask恒等于 None
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            Now_img_txt_token_spans = Now_img_txt_token_spans,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class My_LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        print("---padding_idx---:", self.padding_idx)
        print("---num_hidden_layers---:", config.num_hidden_layers)   # config.num_hidden_layers = 40
        self.layers = nn.ModuleList(
            [My_LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    """!!!"""
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        Now_img_txt_token_spans = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        """!!!"""
        # print("My_LlamaModel---Now_img_txt_token_spans:", Now_img_txt_token_spans)
        """!!!"""
        
        # sprint("---padding_idx---:", self.padding_idx)     # ---padding_idx--- 恒等于 0
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        """!!!"""
        print("self._use_flash_attention_2:", self._use_flash_attention_2)        # self._use_flash_attention_2 = False
        print("self._use_sdpa:", self._use_sdpa)                                  # self._use_sdpa = True
        print("output_attentions:", output_attentions)                            # output_attentions = False
        
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            
        elif self._use_sdpa and not output_attentions:  #执行此句
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        
        """!!!"""
        # print("hidden_states:", hidden_states.size())

    
        """!!!"""
        if attention_mask != None:
            attention_mask = attention_mask.to('cuda')
        # print("attention_mask_2:", attention_mask)
        # print(attention_mask)                                  # 这里attention_mask恒等于 None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            
            # """!!!""" 不注释掉会报错！
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    Now_img_txt_token_spans = Now_img_txt_token_spans
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


class My_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        """!!!"""
        self.model = LlavaLlamaModel(config)
        self.t_vocab_size = config.vocab_size
        self.t_hidden_size = config.hidden_size
        self.t_pad_token_id = config.pad_token_id
        self.t_embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.now_prompt_tokens_embeds = None     # for no image

    """!!!"""
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
        return_dict: Optional[bool] = None,
        Now_img_txt_token_spans = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        """!!!"""
        # print("My_LlamaForCausalLM---Now_img_txt_token_spans:", Now_img_txt_token_spans)
        """!!!"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)




        
        # ----------------------------------------------------------------------------------------------------------------
        # Prepare Data

        if inputs_embeds != None:                                                    # inputs_embeds只在generate的第一次不是None，之后全是None
            print("inputs_embeds.size:", inputs_embeds.size())   
            self.now_prompt_tokens_embeds = inputs_embeds
        else:
            print("inputs_embeds:", inputs_embeds)


        if input_ids != None:                                                        # input_ids只在generate的第一次是None，之后全不是None（与inputs_embeds相反）
            input_ids_embeds = self.t_embed_tokens(input_ids)
            
            new_inputs_embeds = torch.cat((self.now_prompt_tokens_embeds, input_ids_embeds), dim=1)
            self.now_prompt_tokens_embeds = new_inputs_embeds
            
            print("---new_inputs_embeds---", new_inputs_embeds.size())
        else:
            new_inputs_embeds = self.now_prompt_tokens_embeds

        
        new_position_ids = torch.arange(0, new_inputs_embeds.size(-2), dtype=torch.long, device='cuda:0').unsqueeze(0)
        



        # ----------------------------------------------------------------------------------------------------------------
        # First Stage
        
        print("attention_mask_1:", attention_mask.size())
        # print(attention_mask)                                         # 这里的attention_mask为 torch.Size([1, 637])和 torch.Size([1, 638])的全 1矩阵
        
        if past_key_values == None:
            print("#####past_key_values#####", past_key_values)
        else:
            print("#####past_key_values#####", len(past_key_values))    # past_key_values是一个tuple
    
        
        # print("#####inputs_embeds1#####", new_inputs_embeds.size())
        print("#####use_cache#####:", use_cache)                     # use_cache恒等于 True (经测试，use_cache不影响结果)
        outputs_1 = self.model(
            input_ids=None,
            attention_mask=attention_mask,                     # First Stage和 Second Stage仅attention_mask不一样
            position_ids=new_position_ids,
            past_key_values=None,
            inputs_embeds=new_inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            Now_img_txt_token_spans = Now_img_txt_token_spans
        )

        hidden_states_1 = outputs_1[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices_1 = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits_1 = [F.linear(hidden_states_1, lm_head_slices_1[i]) for i in range(self.config.pretraining_tp)]
            logits_1 = torch.cat(logits_1, dim=-1)
        else:
            logits_1 = self.lm_head(hidden_states_1)
        logits_1 = logits_1.float()



        """!!!"""

        
        # print("logits_1 now:", logits_1.size())

        next_token_logit_1 = logits_1[:, -1, :]
        # print("next_token_logit_1", next_token_logit_1.size())                               # torch.Size([1, 32000])

        next_token_id_1 = torch.argmax(next_token_logit_1, dim=-1)  
        # print("next_token_id_greedy_1", next_token_id_1)                                     # tensor([2], device='cuda:0')

        logit_value_1 = next_token_logit_1[0][next_token_id_1[0]]
        # print("logit_value_1", logit_value_1)                                                # tensor(25.7344, device='cuda:0')


        next_token_probs_1 = torch.nn.functional.softmax(next_token_logit_1, dim=-1)
        # print("next_token_probs_1", next_token_probs_1.size())                               # torch.Size([1, 32000])

        probs_value_1 = next_token_probs_1[0][next_token_id_1[0]]
        # print("probs_value_1", probs_value_1)

        
        
        
        # ----------------------------------------------------------------------------------------------------------------
        # Second Stage
        

        print("Now_img_txt_token_spans", Now_img_txt_token_spans)

        now_tokens_length = attention_mask.size(-1)
        attention_mask_no_img = torch.ones((1, now_tokens_length), dtype=torch.int)


        #  TBD: (这样做可以，但不是很好。虽然Now_img_txt_token_spans没更新到最新的长度，但由于只关注image token的范围，最后text tokens的范围不会影响)
        for segment in Now_img_txt_token_spans[0]:      
            token_type, start_idx, end_idx = segment
            if token_type == 'image':
                attention_mask_no_img[0, start_idx:end_idx] = 0  

        # print("attention_mask.shape:", attention_mask.shape)  
        # print("attention_mask_no_img.shape:", attention_mask_no_img.shape)  


        # print("#####inputs_embeds2#####", new_inputs_embeds.size())
        outputs_no_img = self.model(
            input_ids=None,
            attention_mask=attention_mask_no_img,
            position_ids=new_position_ids,
            past_key_values=None,                  # 注意，这里要置为None，不然会沿用之前计算的 K,V,而这些 K,V是没屏蔽image tokens的
            inputs_embeds=new_inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            Now_img_txt_token_spans = Now_img_txt_token_spans
        )
        

        hidden_states_no_img = outputs_no_img[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits_no_img = [F.linear(hidden_states_no_img, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits_no_img = torch.cat(logits_no_img, dim=-1)
        else:
            logits_no_img = self.lm_head(hidden_states_no_img)
            
        logits_no_img = logits_no_img.float()


        """!!!"""

        
        # print("logits_no_img now:", logits_no_img.size())

        next_token_logit_no_img = logits_no_img[:, -1, :]
        # print("next_token_logit_no_img", next_token_logit_no_img.size())                               # torch.Size([1, 32000])

        next_token_id_no_img = torch.argmax(next_token_logit_no_img, dim=-1)  
        # print("next_token_id_no_img_greedy", next_token_id_no_img)                                     # tensor([2], device='cuda:0')

        logit_value_no_img = next_token_logit_no_img[0][next_token_id_no_img[0]]
        # print("logit_value_no_img", logit_value_no_img)                                                # tensor(25.7344, device='cuda:0')

        
        next_token_probs_no_img = torch.nn.functional.softmax(next_token_logit_no_img, dim=-1)
        # print("next_token_probs_no_img", next_token_probs_no_img.size())                                 # torch.Size([1, 32000])

        probs_value_no_img = next_token_probs_no_img[0][next_token_id_no_img[0]]
        # print("probs_value_no_img", probs_value_no_img)  

        # print("Has img:", next_token_probs_1)
        # print("No  img:", next_token_probs_no_img)
        





        
        # ----------------------------------------------------------------------------------------------------------------
        # Final Stage
        """
        if inputs_embeds == None:
            print("#####inputs_embeds3#####", inputs_embeds)
        else:
            print("#####inputs_embeds3#####", inputs_embeds.size())
        """
            
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
            Now_img_txt_token_spans = Now_img_txt_token_spans
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()




        """!!!"""



        # print("logits now:", logits.size())

        next_token_logit = logits[:, -1, :]
        # print("next_token_logit", next_token_logit.size())                               # torch.Size([1, 32000])

        next_token_id = torch.argmax(next_token_logit, dim=-1)  
        # print("next_token_id_greedy", next_token_id)                                     # tensor([2], device='cuda:0')

        logit_value = next_token_logit[0][next_token_id[0]]
        # print("logit_value", logit_value)                                                # tensor(25.7344, device='cuda:0')


        next_token_probs = torch.nn.functional.softmax(next_token_logit, dim=-1)
        # print("next_token_probs", next_token_probs.size())                               # torch.Size([1, 32000])

        probs_value = next_token_probs[0][next_token_id[0]]
        # print("probs_value", probs_value)




        
        """!!!"""


        print("---------------------------------------------")

        print(f"[Has img]next_token_id: {next_token_id_1} | logit_value: {logit_value_1} | probs_value: {probs_value_1}")
        print(f"[No  img]next_token_id: {next_token_id_no_img} | logit_value: {logit_value_no_img} | probs_value: {probs_value_no_img}")
        print(f"[Fin img]next_token_id: {next_token_id} | logit_value: {logit_value} | probs_value: {probs_value}")
        

        print("---------------------------------------------")


        """!!!"""
        # 计算 Delta Probs
        Probs_1 = next_token_probs_1[0][next_token_id_1[0]]
        Probs_2 = next_token_probs_no_img[0][next_token_id_1[0]]

        Delta_Probs = Probs_1 - Probs_2

        print(f"[Delta_Probs]: {Delta_Probs} | Probs_1: {Probs_1} | Probs_2: {Probs_2} |")
        
        print("---------------------------------------------")

        print("sum_probs:", next_token_probs_1[0].sum(), next_token_probs_no_img[0].sum(), next_token_probs[0].sum())

        print("---------------------------------------------")

        

        # ----------------------------------------------------------------------------------------------------------------
        

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        

        # return_dict = True, 不执行此句
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




# -------------------------------------------------------------------------------------------------------------------






class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, My_LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        # print("----------------------------Using CustomLlamaModel----------------------------")
        super().__init__(config)


class LlavaLlamaForCausalLM(My_LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        print("----------------------------Using My_LlamaForCausalLM----------------------------")
        # super(LlamaForCausalLM, self).__init__(config)
        super().__init__(config)
        
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        """!!!"""
        self.Now_img_txt_token_spans = []
        """!!!"""
        

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        """!!!"""
        # print("forward")  # 执行forward函数
        # print("inputs_embeds", inputs_embeds)
        """!!!"""

        """!!!"""  # 注意！！！三引号不能跟在执行语句后用！！！
        img_txt_token_spans = []   # forward中第一次为[],之后全是None,因为输入forward中的token都不含image

        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                img_txt_token_spans
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )


        """!!!"""
        # print("forward-img_txt_token_spans:", img_txt_token_spans)  
        # print("forward---self.Now_img_txt_token_spans:", self.Now_img_txt_token_spans)  

        """!!!"""

        
        """!!!"""
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
            Now_img_txt_token_spans = self.Now_img_txt_token_spans
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        """!!!"""
        # print("generate")   # 只在每次 提问后 一开始执行一次
        """!!!"""
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")


        """!!!"""
        img_txt_token_spans = None
        if images is not None:
            print("Has image!")
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                img_txt_token_spans
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            print("No image!")
            img_txt_token_spans = [[['text', 0, inputs.size(1)]]]
            inputs_embeds = self.get_model().embed_tokens(inputs)


        """!!!"""
        # print("generate-img_txt_token_spans:", img_txt_token_spans)  # 这里会获取正确的image和question tokens的范围
        
        self.Now_img_txt_token_spans = img_txt_token_spans
        
        print("---self.Now_img_txt_token_spans:", self.Now_img_txt_token_spans)
        
        """!!!"""
        
        # print(inspect.getsource(super().generate))
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
