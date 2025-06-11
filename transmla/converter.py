import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from modify_config import modify_config
from utils import get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl
from partial_rope import partial_rope
from lora_qkv import low_rank_qkv


def load_model_and_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        device_map=args.device,
        _attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_dataset_loader(tokenizer: AutoTokenizer, **kwargs):
    dataset = get_dataset(kwargs["cal_dataset"])
    train_loader = prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=kwargs["cal_max_seqlen"],
        batch_size=kwargs["cal_batch_size"],
        nsamples=kwargs["cal_nsamples"],
        seed=kwargs["seed"],
    )
    if kwargs["ppl_eval_batch_size"] > 0:
        test_loader = prepare_test_dataloader(
            dataset=dataset["test"], tokenizer=tokenizer, batch_size=kwargs["ppl_eval_batch_size"]
        )
    else:
        test_loader = None
    
    return train_loader, test_loader

    
def main(args):

    ##############################
    #       original model       #
    ##############################
    print("\n" + "="*60)
    print("Original Model".center(60))
    print("="*60 + "\n")

    # get model, tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    # get dataset
    train_loader, test_loader = get_dataset_loader(tokenizer, **vars(args))

    # if test_loader:
    #     message = "Evaluating original model's ppl"
    #     dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
    #     print(f'Original ppl: {dataset_ppl:.4f}')

    ##############################
    #        partial rope        #
    ##############################
    print("\n" + "="*60)
    print("Partial RoPE Model".center(60))
    print("="*60 + "\n")

    if args.collapse == "auto":
        head_dim = model.config.head_dim if hasattr(model.config, "head_dim") and model.config.head_dim is not None else model.config.hidden_size // model.config.num_attention_heads
        model.config.head_dim = head_dim
        args.collapse = head_dim // args.qk_mqa_dim
        print(f"Auto collapse: {args.collapse} (head_dim={head_dim} / qk_mqa_dim={args.qk_mqa_dim})")
    else:
        args.collapse = int(args.collapse)

    model = partial_rope(model, tokenizer, train_loader, test_loader, **vars(args))
    if args.freqfold == "auto":
        args.freqfold = model[1]
        model = model[0]

    ##############################
    #     deepseek-mla model     #
    ##############################
    print("\n" + "="*60)
    print("LoraQKV Model".center(60))
    print("="*60 + "\n")

    model = low_rank_qkv(model, tokenizer, train_loader, test_loader, **vars(args))

    # save model
    print(f"\nSaving model and tokenizer to {args.save_path}...")
    model.save_pretrained(os.path.join(args.save_path))
    tokenizer.save_pretrained(os.path.join(args.save_path))

    # modify config
    modify_config(model, os.path.join(args.save_path, "config.json"), args)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model to load")
    parser.add_argument("--save-path", type=str, default="outputs", help="output path.")
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, help="Device to use.", default="auto")
    parser.add_argument("--cal-dataset", type=str, help="Dataset to calibrate and calculate perplexity on.", choices=["wikitext2", "ptb", "c4", "alpaca"], default="wikitext2")
    parser.add_argument("--cal-nsamples", type=int, help="Number of samples of the calibration data to load.", default=128)
    parser.add_argument("--cal-batch-size", type=int, default=8, help="Batch size for loading the calibration data.")
    parser.add_argument("--cal-max-seqlen", type=int, default=256, help="Maximum sequence length for the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--ppl-eval-batch-size", type=int, default=0, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--freqfold", type=str, default="auto", help="Freqfold for removing RoPE, int or auto")
    parser.add_argument("--collapse", type=str, default="auto", help="Collapse for removing RoPE, int or auto")
    parser.add_argument("--qk-mqa-dim", type=int, default=64, help="")
    parser.add_argument("--q-lora-rank", type=int, help="")
    parser.add_argument("--kv-lora-rank", type=int, default=512, help="")
    parser.add_argument("--balance-kv-ratio", type=float, default=1, help="")
    parser.add_argument("--use-qkv-norm", action='store_true', default=False, help="")
    args = parser.parse_args()

    main(args)
