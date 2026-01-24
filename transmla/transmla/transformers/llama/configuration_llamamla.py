from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaMLAConfig(LlamaConfig):
    model_type = "deepseek_v3"

    def __init__(
        self, 
        *args, 
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        qk_latent_layernorm=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_latent_layernorm = qk_latent_layernorm