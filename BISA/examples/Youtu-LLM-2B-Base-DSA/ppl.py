"""
Perplexity (PPL) evaluation script for the local model.
Usage:
    python ppl.py [--dataset DATASET] [--split SPLIT] [--max_length MAX_LENGTH] [--stride STRIDE] [--batch_size BATCH_SIZE]
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a causal language model.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model directory. Defaults to the script's directory.")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="HuggingFace dataset name (default: wikitext)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset config name (default: wikitext-2-raw-v1)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max sequence length for evaluation (default: 2048)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding window (default: 512)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    return parser.parse_args()


def load_dataset_text(dataset_name, dataset_config, split):
    """Load dataset and concatenate all text."""
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    # Concatenate all text with double newlines
    text = "\n\n".join(dataset["text"])
    return text


def evaluate_ppl(model, tokenizer, text, max_length, stride, device, batch_size=1):
    """
    Evaluate perplexity using a sliding window approach.
    This follows the standard approach described in the HuggingFace documentation.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    print(f"Total tokens in dataset: {seq_len}")
    print(f"Max length: {max_length}, Stride: {stride}")

    nlls = []
    prev_end_loc = 0
    num_windows = (seq_len - 1) // stride + 1

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # number of new tokens to evaluate

        chunk_input_ids = input_ids[:, begin_loc:end_loc]
        target_ids = chunk_input_ids.clone()
        # Mask out the tokens we've already computed loss for (overlap region)
        target_ids[:, :-trg_len] = -100

        # Build position_ids (required by this model)
        batch_size_actual = chunk_input_ids.size(0)
        chunk_seq_len = chunk_input_ids.size(1)
        position_ids = torch.arange(chunk_seq_len, device=device).unsqueeze(0).expand(batch_size_actual, -1)

        with torch.no_grad():
            outputs = model(
                input_ids=chunk_input_ids,
                labels=target_ids,
                position_ids=position_ids,
            )
            # The model computes loss internally via LigerFusedLinearCrossEntropyLoss
            # We need to recompute per-token NLL for accurate PPL
            neg_log_likelihood = outputs.loss

        # The loss is averaged over the valid (non -100) tokens
        # We need to scale it back to get the total NLL for the new tokens
        # Count valid tokens in target_ids
        num_valid_tokens = (target_ids != -100).sum().item()
        # Subtract 1 because the model shifts labels internally (labels[..., 1:])
        # So the actual number of tokens the loss is computed over may differ
        # We store (loss * num_valid_tokens_after_shift) to accumulate total NLL
        # After shift: target_ids[:, 1:] compared with logits[:, :-1]
        shifted_targets = target_ids[:, 1:]
        num_loss_tokens = (shifted_targets != -100).sum().item()

        if num_loss_tokens > 0:
            nlls.append(neg_log_likelihood.float() * num_loss_tokens)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    # Total number of tokens that were evaluated
    # This equals (seq_len - 1) since each token (except the first) is predicted exactly once
    total_tokens = seq_len - 1
    avg_nll = total_nll / total_tokens
    ppl = torch.exp(avg_nll)

    return ppl.item(), avg_nll.item(), total_tokens


def main():
    args = parse_args()

    # Default model path is the directory containing this script
    if args.model_path is None:
        args.model_path = os.path.dirname(os.path.abspath(__file__))

    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}/{args.dataset_config} (split: {args.split})")
    print(f"Device: {args.device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Load dataset
    print("Loading dataset...")
    text = load_dataset_text(args.dataset, args.dataset_config, args.split)

    # Evaluate
    print("Starting PPL evaluation...")
    ppl, avg_nll, total_tokens = evaluate_ppl(
        model, tokenizer, text,
        max_length=args.max_length,
        stride=args.stride,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("=" * 60)
    print(f"Dataset:       {args.dataset}/{args.dataset_config} ({args.split})")
    print(f"Max Length:     {args.max_length}")
    print(f"Stride:        {args.stride}")
    print(f"Total Tokens:  {total_tokens}")
    print(f"Avg NLL:       {avg_nll:.4f}")
    print(f"Perplexity:    {ppl:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
