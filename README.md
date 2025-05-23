# TransMLA: Equivalently Transforms Group Query Attention into Multi-Head Latent Attention.

Modern large language models (LLMs) often encounter communication bottlenecks, rather than computational ones, on current hardware. Multi-head Latent Attention (MLA) addresses these constraints by employing low-rank matrices for KV layers, allowing for caching of compressed latent KV states. This substantially reduces the activation cache size compared to standard multi-head attention and accelerates inference. Moreover, MLA incorporates an up-projection matrix for enhanced expressiveness, effectively trading additional computation for reduced communication overhead.

Despite the proven efficiency and effectiveness of MLA in DeepseekV2/V3, major model providers still rely on GQA, with no public plans to transition to MLA. To facilitate broader adoption, we introduce TransMLA, a post-training method that converts widely used GQA-based pre-trained models into MLA models. This conversion is followed by further training to boost expressiveness without increasing the KV cache size. We also plan to develop MLA-specific inference acceleration techniques to reduce the transformed model's inference latency, ultimately unlocking MLA’s full potential in large-scale LLM deployments.

# News
- [2025.04.28] Released TransMLA v3, successfully apply PCA across RoPE and reduce KV Cache.
- [2025.02.16] Released the second version of the TransMLA model and usage code, compatible with RoPE and supporting Absorb operation.
- [2025.02.13] The technical report of TransMLA is publicly available: [https://huggingface.co/papers/2502.07864](https://huggingface.co/papers/2502.07864)
- [2025.01.02] Released the first version of the TransMLA model code, providing usage code for converting Qwen2.5 and LLaMA-3’s GQA to MLA equivalence.

# Install
```
conda create -n transmla python=3.12.8
conda activate transmla
pip install vllm==0.8.4
pip install datasets
pip install accelerate==1.3.0
pip install ipykernel
pip install datatrove
pip install tensorboardX
```

# Run
```
python main.py
```

# To-Do
- [ ] Publish the technical report for the new version, detailing how TransMLA is compatible with RoPE, supports the Absorb operation.
- [x] Compress the dimensions of the KV cache to improve inference speed.
- [ ] Release checkpoint.
- [x] Add support for vLLM to improve inference speed.
- [x] Support FlashMLA.
- [ ] Extend support to additional models (e.g., LLaMA, Mistral, Gemma2, etc.).
- [ ] Fine-tune on R1 distillation datasets.

# Citation
```
@article{meng2025transmla,
  title={TransMLA: Multi-head Latent Attention Is All You Need},
  author={Meng, Fanxu and Tang, Pingzhi and Yao, Zengwei and Zhang, Muhan},
  journal={arXiv preprint arXiv:2502.07864},
  year={2025}
}
```
