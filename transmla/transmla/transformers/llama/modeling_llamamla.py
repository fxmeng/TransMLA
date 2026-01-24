from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaForCausalLM
)

from .configuration_llamamla import LlamaMLAConfig
from .mla import MLAAttention, eager_attention_forward


class LlamaMLADecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: LlamaMLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MLAAttention(config, layer_idx)


class LlamaMLAPreTrainedModel(LlamaPreTrainedModel):

    config_class = LlamaMLAConfig
    _no_split_modules = ["LlamaMLADecoderLayer"]


class LlamaMLAModel(LlamaMLAPreTrainedModel, LlamaModel):

    def __init__(self, config: LlamaMLAConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [LlamaMLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class LlamaMLAForCausalLM(LlamaMLAPreTrainedModel, LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaMLAModel(config)


__all__ = [
    "LlamaMLAForCausalLM",
    "LlamaMLAModel",
    "LlamaMLAPreTrainedModel",
]