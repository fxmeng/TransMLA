import json
import transformers.models as models


settings = {
    "llama": {
        "config_class": models.llama.configuration_llama.LlamaConfig,
        "auto_map": {
            "AutoConfig": "configuration_llamamla.LlamaMLAConfig",
            "AutoModel": "modeling_llamamla.LlamaMLAModel",
            "AutoModelForCausalLM": "modeling_llamamla.LlamaMLAForCausalLM"
        },
        "architectures": ["LlamaMLAForCausalLM"],
        "model_type": "llamamla",
    },
    "mixtral": {
        "config_class": models.mixtral.configuration_mixtral.MixtralConfig,
        "auto_map": {
            "AutoConfig": "configuration_mixtralmla.MixtralMLAConfig",
            "AutoModel": "modeling_mixtralmla.MixtralMLAModel",
            "AutoModelForCausalLM": "modeling_mixtralmla.MixtralMLAForCausalLM"
        },
        "architectures": ["MixtralMLAForCausalLM"],
        "model_type": "mixtralmla",
    }, 
    "gemma2": {
        "config_class": models.gemma2.configuration_gemma2.Gemma2Config,
        "auto_map": {
            "AutoConfig": "configuration_gemma2mla.Gemma2MLAConfig",
            "AutoModel": "modeling_gemma2mla.Gemma2MLAModel",
            "AutoModelForCausalLM": "modeling_gemma2mla.Gemma2MLAForCausalLM"
        },
        "architectures": ["Gemma2MLAForCausalLM"],
        "model_type": "gemma2mla",
    }
}
settings["qwen2"] = settings["llama"]
settings["mistral"] = settings["llama"]



def modify_config(model, config_path: str, args):
    import json

    with open(config_path, "r") as f:
        config = json.load(f)
    setting = settings[model.config.model_type]

    config["auto_map"] = setting["auto_map"]
    config["architectures"] = setting["architectures"]
    config["model_type"] = setting["model_type"]
    
    config["num_key_value_heads"] = config["num_attention_heads"]
    config["attention_bias"] = model.model.layers[0].self_attn.attention_bias
    config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
    config["qk_nope_head_dim"] = config["v_head_dim"] = model.model.layers[0].self_attn.head_dim
    config["q_lora_rank"] = args.q_lora_rank
    config["kv_lora_rank"] = args.kv_lora_rank

    if config.get("query_pre_attn_scalar", None) is None:
        config["query_pre_attn_scalar"] = model.model.layers[0].self_attn.head_dim

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)