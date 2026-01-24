from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs

from transformers.models.mixtral.modeling_mixtral import (
    MixtralModel,
    MixtralDecoderLayer,
    MixtralPreTrainedModel,
    MixtralForCausalLM
)

from .configuration_mixtralmla import MixtralMLAConfig
from .mla import MLAAttention, eager_attention_forward


class MixtralMLADecoderLayer(MixtralDecoderLayer):

    def __init__(self, config: MixtralMLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MLAAttention(config, layer_idx)


class MixtralMLAPreTrainedModel(MixtralPreTrainedModel):

    config_class = MixtralMLAConfig
    _no_split_modules = ["MixtralMLADecoderLayer"]


class MixtralMLAModel(MixtralMLAPreTrainedModel, MixtralModel):

    def __init__(self, config: MixtralMLAConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [MixtralMLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class MixtralMLAForCausalLM(MixtralMLAPreTrainedModel, MixtralForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = MixtralMLAModel(config)


__all__ = [
    "MixtralMLAForCausalLM",
    "MixtralMLAModel",
    "MixtralMLAPreTrainedModel",
]