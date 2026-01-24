import json
import transformers.models as models
import shutil
import os


settings = {
    "llama": {
        "auto_map": {
            "AutoConfig": "configuration_llamamla.LlamaMLAConfig",
            "AutoModel": "modeling_llamamla.LlamaMLAModel",
            "AutoModelForCausalLM": "modeling_llamamla.LlamaMLAForCausalLM"
        },
        "architectures": ["LlamaMLAForCausalLM"],
    },
    "mixtral": {
        "auto_map": {
            "AutoConfig": "configuration_mixtralmla.MixtralMLAConfig",
            "AutoModel": "modeling_mixtralmla.MixtralMLAModel",
            "AutoModelForCausalLM": "modeling_mixtralmla.MixtralMLAForCausalLM"
        },
        "architectures": ["MixtralMLAForCausalLM"],
    }, 
    "gemma2": {
        "auto_map": {
            "AutoConfig": "configuration_gemma2mla.Gemma2MLAConfig",
            "AutoModel": "modeling_gemma2mla.Gemma2MLAModel",
            "AutoModelForCausalLM": "modeling_gemma2mla.Gemma2MLAForCausalLM"
        },
        "architectures": ["Gemma2MLAForCausalLM"],
    },
    "deepseek_v3": {
        "auto_map": {
            "AutoConfig": "configuration_deepseek_v3.DeepseekV3Config",
            "AutoModel": "modeling_deepseek_v3.DeepseekV3Model",
            "AutoModelForCausalLM": "modeling_deepseek_v3.DeepseekV3ForCausalLM"
        },
        "architectures": ["DeepseekV3ForCausalLM"],
    }
}
settings["qwen2"] = settings["llama"]
settings["mistral"] = settings["llama"]
settings["mimo"] = settings["llama"]

transformers_dirs = {
    "llama": "transmla/transformers/llama", 
    "gemma2": "transmla/transformers/gemma2", 
    "mixtral": "transmla/transformers/mixtral", 
    "deepseek_v3": "transmla/transformers/deepseek_v3", 
}
transformers_dirs["qwen2"] = transformers_dirs["llama"]
transformers_dirs["mistral"] = transformers_dirs["llama"]
mla_dir = "transmla/transformers/mla.py"



def modify_config(model, config_path: str, args):
    import json

    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = "deepseek_v3" if args.deepseek_style else model.config.model_type
    setting = settings[model_type]

    for key, value in setting.items():
        config[key] = value
    
    config["model_type"] = "deepseek_v3"
    config["num_key_value_heads"] = config["num_attention_heads"]
    config["attention_bias"] = model.model.layers[0].self_attn.attention_bias
    config["qk_rope_head_dim"] = config["head_dim"] = args.qk_mqa_dim
    config["qk_nope_head_dim"] = config["v_head_dim"] = model.model.layers[0].self_attn.head_dim
    config["q_lora_rank"] = args.q_lora_rank
    config["kv_lora_rank"] = args.kv_lora_rank

    config["qk_latent_layernorm"] = hasattr(model.model.layers[0].self_attn, "kv_a_layernorm")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # copy transformers files to the saving path
    transformers_dir = transformers_dirs[model_type]
    for item in os.listdir(transformers_dir):
        source_path = os.path.join(transformers_dir, item)
        shutil.copy(source_path, args.save_path)
    shutil.copy(mla_dir, args.save_path)