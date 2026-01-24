from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs

from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Model,
    Gemma2DecoderLayer,
    Gemma2PreTrainedModel,
    Gemma2ForCausalLM
)

from .configuration_gemma2mla import Gemma2MLAConfig
from .mla import MLAAttention, eager_attention_forward


class Gemma2MLADecoderLayer(Gemma2DecoderLayer):

    def __init__(self, config: Gemma2MLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MLAAttention(config, layer_idx)


class Gemma2MLAPreTrainedModel(Gemma2PreTrainedModel):

    config_class = Gemma2MLAConfig
    _no_split_modules = ["Gemma2MLADecoderLayer"]


class Gemma2MLAModel(Gemma2MLAPreTrainedModel, Gemma2Model):

    def __init__(self, config: Gemma2MLAConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [Gemma2MLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Gemma2MLAForCausalLM(Gemma2MLAPreTrainedModel, Gemma2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2MLAModel(config)


__all__ = [
    "Gemma2MLAForCausalLM",
    "Gemma2MLAModel",
    "Gemma2MLAPreTrainedModel",
]