# 🚀 TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup

Modern large-language models often face communication bottlenecks on current hardware rather than computational limitations. Multi-head latent attention (MLA) addresses this by compressing the key-value cache using low-rank matrices, while the Absorb operation prevents the KV cache from reverting to its original size, significantly boosting both training and inference speed. 

Despite the success of DeepSeek V2/V3/R1, most model vendors have heavily invested in optimizing GQA-based models and therefore lack strong incentives to retrain MLA-based models from scratch. In this paper, we introduce TransMLA, a framework that seamlessly converts any GQA-based pre-trained model (e.g., LLaMA, Qwen, Mixtral) into an MLA-based model. 


# 📰 News
- [2025.05.29] A new version of technical report is released: [https://arxiv.org/abs/2502.07864](https://arxiv.org/abs/2502.07864).
- [2025.04.28] Released TransMLA v3, successfully apply PCA across RoPE and reduce KV Cache.
- [2025.02.16] Released the second version of the TransMLA model and usage code, compatible with RoPE and supporting Absorb operation.
- [2025.02.13] The technical report of TransMLA is publicly available: [https://huggingface.co/papers/2502.07864](https://huggingface.co/papers/2502.07864)
- [2025.01.02] Released the first version of the TransMLA model code, providing usage code for converting Qwen2.5 and LLaMA-3’s GQA to MLA equivalence.

# 🛠 Installation
```
git clone https://github.com/fxmeng/TransMLA.git
cd TransMLA
conda create -n transmla python=3.12.8
conda activate transmla
pip install -r requirements.txt
```

# ⚡ Quick Start

1. Convert MHA / GQA models (e.g. Qwen2.5-7B-Instruct) into DeepSeek-MLA:
    ```bash
    bash scripts/convert/qwen2.5-7B-Instruct.sh
    ```
2. Have fun playing with the converted models!
    ```python
    # using `Transformers.AutoModelForCausalLM`
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("outputs/qwen2_5-7B-Instruct-deepseek", trust_remote_code=True)

    # using `vllm.LLM`
    # note that only Llama-type models(llama, qwen, mistral) are supported right now
    import transmla.vllm_registry.deepseek      # register mla models
    from vllm import LLM, SamplingParams
    llm = LLM(model="outputs/qwen2_5-7B-Instruct-deepseek", trust_remote_code=True)
    ```

## 🔧 Advanced Usage (`converter.py`)

The converter.py script allows you to perform fine-grained control over RoPE removal and low-rank QKV projection towards DeepSeek-MLA. It supports:
- Auto-search for optimal freqfold that minimizes PPL.
- Automatic computation of collapse based on head_dim / qk_mqa_dim.
- Evaluation of original, RoPE-removed, and final MLA models.


### ✅ Example Command:
```bash
python transmla/converter.py \
    --model-path meta-llama/Llama-2-7b-hf \
    --save-path ./outputs/llama2-7b-deepseek \
    --dtype bf16 \
    --device auto \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-max-seqlen 256 \
    --cal-batch-size 8 \
    --ppl-eval-batch-size 4 \
    --freqfold auto \
    --collapse auto \
    --qk-mqa-dim 64 \
    --q-lora-rank 512 \
    --kv-lora-rank 512
```

### 📘 Argument Details

| Argument | Description |
|----------|-------------|
| --model-path | Path to the base model (e.g., from HuggingFace hub). |
| --save-path | Output path for the converted model and tokenizer. |
| --cal-dataset | Calibration dataset: wikitext2, ptb, c4, or alpaca. |
| --cal-nsamples, --cal-max-seqlen, --cal-batch-size | Number, max sequence length, and batch size of samples used for calibration. |
| --freqfold | RoPE frequency folding factor, or `auto` to search for the best value. Note: Automatic freqfold search is only supported in single-GPU setups currently. Please set the device explicitly, for example: `cuda:0`. |
| --collapse | Collapse factor for RoPE. Use `auto` to compute as `head_dim // qk_mqa_dim`. Collapse factor reduces the dim of RoPEd KV cache from `head_dim` to `head_dim // collapse`. |
| --qk-mqa-dim | Target dimension for decoupled RoPE. |
| --q-lora-rank | The inner dimension for query low-rank decomposition, or `None` to disable low-rank decomposition for query. |
| --kv-lora-rank | The inner dimension for key/value joint low-rank decomposition. |
| --deepseek-style | Use deepseek style modeling / configuration files from transformers. Only support Llama-type models(llama, qwen, mistral)


### 🧠 Tips
- Set `--freqfold auto` and `--collapse auto` to simplify configuration. The script will automatically search for the best freqfold factor based on ppl results.
- We recommend setting `--qk-mqa-dim` to 64 and `--kv-lora-rank` to 512 to satisfy FlashMLA's requirements on H100.


# 🐒 Model Zoo

| Model Family | Model | kv-lora-rank + qk-mqa-dim | freqfold | Original ppl | Partial RoPE ppl | MLA ppl |
| - | - | - | - | - | - | - |
| Llama2    | Llama-2-7B            | 512 + 64   | 8 | 5.4732 | 18.6373 | 41.6135 |
|           |                       | 448 + 128  | 8 |        | 8.9903  | 25.7731 |
| Llama3    | Llama-3-8B            | 512 + 64   | 4 | 6.1371 | 12.0550 | 25.8047 |
|           |                       | 448 + 128  | 4 |        | 8.3997  | 18.3500 |
|           | Llama-3.2-1B          | 512 + 64   | 4 | 9.7531 | 16.3391 | 16.1404 |
| Qwen2     | Qwen2.5-7B            | 512 + 64   | 4 | 6.8480 | 7.8448  | 8.4124  |
|           |                       | 448 + 128  | 4 |        | 7.3059  | 7.9812  |
|           | Qwen2.5-7B-Instruct   | 512 + 64   | 4 | 7.4570 | 8.8902  | 10.0082 |
|           |                       | 448 + 128  | 4 |        | 8.0734  | 9.1957  |
|           | Qwen2.5-72B-Instruct  | 512 + 64   | 4 | 4.2687 | 4.9650  | 7.3850  |
|           |                       | 448 + 128  | 4 |        | 4.6931  | 7.7172  |
| Gemma2    | gemma-2-9b-it         | 512 + 64   | 8 | 10.1612| 11.4207 | 21.6260 |
|           |                       | 448 + 128  | 8 |        | 10.9948 | 22.0038 |
|           |                       | 320 + 256  | 4 |        | 10.7075 | 32.4387 |
| Mistral   | Mistral-7B-v0.3       | 512 + 64   | 8 | 5.3178 | 7.4697  | 9.5830  |
|           |                       | 448 + 128  | 8 |        | 5.5915  | 7.0251  |
| Mixtral   | Mixtral-8x7B-v0.1     | 512 + 64   | 8 | 3.8422 | 5.6310  | 7.5179  |
|           |                       | 448 + 128  | 4 |        | 4.1407  | 5.8374  |
| MiMo      | MiMo-7B-Base          | 512 + 64   | 4 | 6.9108 | 7.9272  | 9.5810  |


# 📋 To-Do
- [x] Publish the technical report for the new version, detailing how TransMLA is compatible with RoPE, supports the Absorb operation.
- [x] Compress the dimensions of the KV cache to improve inference speed.
- [x] Add support for vLLM to improve inference speed.
- [x] Support FlashMLA.
- [x] Extend support to additional models (e.g., LLaMA, Mistral, Gemma2, etc.).
- [ ] Support GTA & GLA
- [ ] Release checkpoints.
- [ ] Fine-tune on R1 distillation datasets.


# 📚 Citation
```
@article{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Tang, Pingzhi and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864},
  year={2025}
}
```

# ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=fxmeng/TransMLA&type=Date)](https://www.star-history.com/#fxmeng/TransMLA&Date)
