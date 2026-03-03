from typing import Literal
import warnings
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config as HFDeepseekV3Config

class DeepseekV3Config(HFDeepseekV3Config):
    model_type = "deepseek_v3"
    def __init__(
        self,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = None,
        qk_latent_norm: bool = True,
        indexer_norm: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        index_weights: Literal["value", "one"] = "value",
        index_absorb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.qk_latent_norm = qk_latent_norm
        self.indexer_norm = indexer_norm
        self.index_weights = index_weights
        self.index_absorb = index_absorb


