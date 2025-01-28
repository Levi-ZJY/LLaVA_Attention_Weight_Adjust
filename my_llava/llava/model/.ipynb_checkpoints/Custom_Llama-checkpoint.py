
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention
from transformers.cache_utils import Cache

# 自定义 LlamaAttention (继承原版)
class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        outputs = super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

        return outputs

        
# 自定义 LlamaDecoderLayer (继承原版)
class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        outputs = super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

        return outputs



class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


















