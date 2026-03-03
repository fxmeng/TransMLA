import math
import torch
import torch.nn.functional as F
from einops import einsum

import tilelang as tl
import tilelang.language as T
from typing import Optional
from index import prepare_token_indices

from util import get_abs_err, get_err_ratio

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def indexer_topk_reducesum_interface(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    offsets: torch.Tensor,
    block_size: int = 128,
    block_topk: int = 64,
    dtype: str = BF16,
    chunk_size: int = 2048,
):
    total_seq_len = q.shape[0]
    device = q.device
    softmax_scale = q.shape[-1] ** -0.5
    
    all_topk_indices = torch.full((total_seq_len, topk), -1, dtype=torch.int64, device=device)
    all_topk_score = torch.full((total_seq_len, topk), float('-inf'), dtype=torch.float32, device=device)
    
    for batch_idx in range(offsets.shape[0] - 1):
        start_idx = offsets[batch_idx].item()
        end_idx = offsets[batch_idx + 1].item()
        seq_len = end_idx - start_idx
        
        q_batch = q[start_idx:end_idx]
        weights_batch = weights[start_idx:end_idx]
        k_batch = k[start_idx:end_idx]
        
        num_blocks = math.ceil(seq_len / block_size)
        pad_len = num_blocks * block_size - seq_len
        if pad_len > 0:
            k_batch_padded = F.pad(k_batch, (0, 0, 0, pad_len), value=0.0)  # [num_blocks * block_size, D]
            block_counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=device)
            block_counts[-1] = block_size - pad_len  # last block has fewer tokens
        else:
            k_batch_padded = k_batch
            block_counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=device)
        k_block_mean = (
            k_batch_padded.reshape(num_blocks, block_size, -1)
            .sum(dim=1)  # [num_blocks, D]
            / block_counts.unsqueeze(-1)
        ).to(k.dtype)  # [num_blocks, D]

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            
            q_chunk = q_batch[chunk_start:chunk_end] # [chunk_len, H, D]
            weights_chunk = weights_batch[chunk_start:chunk_end] # [chunk_len, H]
            chunk_end_block = math.ceil(chunk_end / block_size) 
            k_block_mean_visible = k_block_mean[:chunk_end_block] # [chunk_end_block, D]           

            # Step 1: Compute block-level scores [chunk_len, chunk_end_block]
            block_logits = einsum(q_chunk, k_block_mean_visible, 'cl h d, nb d -> cl h nb')
            block_logits = F.relu(block_logits)
            block_scores = (block_logits * weights_chunk.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale # [chunk_len, chunk_end_block]

            q_positions = torch.arange(chunk_start, chunk_end, device=device)  # [chunk_len]
            block_starts = torch.arange(chunk_end_block, device=device) * block_size  # [chunk_end_block]
            causal_block_mask = (q_positions.unsqueeze(1) >= block_starts.unsqueeze(0))  # [chunk_len, chunk_end_block]
            block_scores = torch.where(causal_block_mask, block_scores, torch.tensor(float('-inf'), device=device))

            # Step 2: Block topk selection with mandatory constraint
            q_block_ids = q_positions // block_size
            mandatory_last1 = q_block_ids
            mandatory_last2 = (q_block_ids - 1).clamp(min=0)

            LARGE_SCORE = 1e9
            batch_indices = torch.arange(chunk_len, device=device)
            block_scores_for_topk = block_scores.clone()
            block_scores_for_topk[:, 0] = LARGE_SCORE
            block_scores_for_topk[batch_indices, mandatory_last1] = LARGE_SCORE
            block_scores_for_topk[batch_indices, mandatory_last2] = LARGE_SCORE

            actual_btopk = min(block_topk, chunk_end_block)
            _, selected_block_indices = torch.topk(block_scores_for_topk, k=actual_btopk, dim=-1) # [chunk_len, actual_btopk]


            # Step 3: Expand selected blocks into token indices [chunk_len, block_topk * block_size]
            block_start_tokens = selected_block_indices * block_size  # [chunk_len, actual_btopk]
            offsets_in_block = torch.arange(block_size, device=device)  # [block_size]
            candidate_indices = (block_start_tokens.unsqueeze(-1) + offsets_in_block).reshape(chunk_len, -1)  # [chunk_len, actual_btopk * block_size]

            valid_mask = (candidate_indices <= q_positions.unsqueeze(1)) & (candidate_indices < seq_len)
            token_indices = torch.where(valid_mask, candidate_indices, torch.tensor(-1, dtype=torch.int64, device=device))  # [chunk_len, actual_btopk * block_size]

            # Pad to full width if actual_btopk < block_topk
            if token_indices.shape[1] < block_topk * block_size:
                token_indices = F.pad(token_indices, (0, block_topk * block_size - token_indices.shape[1]), value=-1)

            # Step 4: Compute token-level scores for selected tokens
            num_candidates = token_indices.shape[1]  # block_topk * block_size
            safe_indices = token_indices.clamp(min=0)  # [chunk_len, num_candidates], replace -1 with 0 for gather
            k_selected = k_batch[safe_indices]  # [chunk_len, num_candidates, D]
            token_logits = einsum(q_chunk, k_selected, 'cl h d, cl nc d -> cl h nc')
            token_logits = F.relu(token_logits)
            token_scores = (token_logits * weights_chunk.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale  # [chunk_len, num_candidates]
            token_scores = torch.where(token_indices >= 0, token_scores, torch.tensor(float('-inf'), device=device))

            # Step 5: Topk token selection and store into all_topk_indices/all_topk_score
            actual_topk = min(topk, num_candidates)
            topk_scores, topk_local_ids = torch.topk(token_scores, k=actual_topk, dim=-1)  # [chunk_len, actual_topk]
            topk_token_indices = torch.gather(token_indices, dim=1, index=topk_local_ids)  # [chunk_len, actual_topk]

            # Keep batch-local indices (consistent with ref_index_score)
            topk_global_indices = torch.where(topk_token_indices >= 0, topk_token_indices, torch.tensor(-1, dtype=torch.int64, device=device))
            topk_final_scores = F.softmax(topk_scores, dim=-1, dtype=torch.float32)
            topk_final_scores = torch.where(topk_token_indices >= 0, topk_final_scores, torch.tensor(float('-inf'), device=device))

            # Store results
            global_start = start_idx + chunk_start
            global_end = start_idx + chunk_end
            if actual_topk < topk:
                all_topk_indices[global_start:global_end, :actual_topk] = topk_global_indices
                all_topk_score[global_start:global_end, :actual_topk] = topk_final_scores
            else:
                all_topk_indices[global_start:global_end] = topk_global_indices
                all_topk_score[global_start:global_end] = topk_final_scores

    return all_topk_indices, all_topk_score


def ref_index_score(Q: torch.Tensor, Weights: torch.Tensor, K: torch.Tensor, topk: int,
                    offsets: torch.Tensor, block_size: int = 128, block_topk: int = 64):
    all_topk_indices = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i]:offsets[i + 1]]
        weights = Weights[offsets[i]:offsets[i + 1]]
        k = K[offsets[i]:offsets[i + 1]]
        softmax_scale = q.shape[-1]**-0.5
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = einsum(q, k, 's1 h k, s2 k -> s1 h s2')
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float('-inf'))
        
        if s < topk:
            pad_size = topk - s
            logits = F.pad(logits, (0, pad_size), value=float('-inf'))
        
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        
        if s < topk:
            valid_mask = topk_indices < s
            topk_indices = torch.where(valid_mask, topk_indices, torch.tensor(-1, dtype=torch.int64, device=q.device))
            topk_score = torch.where(valid_mask, topk_score, torch.tensor(float('-inf'), device=q.device))
        
        all_topk_indices.append(topk_indices)
        all_topk_score.append(topk_score)
    topk_indices = torch.cat(all_topk_indices, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return topk_indices, topk_score


def test_kernel(
    B=4,
    S=19384,
    H=16,
    D=128,
    topk=2048,
):
    from triton.testing import do_bench
    
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    q = torch.randn((S, H, D)).cuda().bfloat16()
    weights = torch.randn((S, H)).cuda().bfloat16()
    k = torch.randn((S, D)).cuda().bfloat16()
    offsets = torch.tensor([0, 1000, 2000, 3000, 19384], dtype=torch.int32).cuda()

    ref_topk_indices, ref_topk_score = ref_index_score(q, weights, k, topk, offsets)

    topk_indices, topk_score = indexer_topk_reducesum_interface(q, weights, k, topk, offsets)

    for j in range(S):
        ref_np = ref_topk_indices[j].cpu().to(torch.int32).numpy()
        trt_np = topk_indices[j].cpu().to(torch.int32).numpy()

        ref_np_val = ref_topk_score[j]
        trt_np_val = topk_score[j]

        mask = (ref_np_val > 0).cpu().numpy()

        set_ref = set(ref_np[mask])
        set_trt = set(trt_np[mask])
        intersection = set_ref & set_trt

        print("idx:", j, "selected/all:", len(intersection), "/", len(set_ref), "=",
              len(intersection) / len(set_ref))

        print(
            f"err: {get_abs_err(ref_np_val, trt_np_val):.6f} ratio: {get_err_ratio(ref_np_val, trt_np_val):.6f}"
        )

    
    # def fn():
    #     return indexer_topk_reducesum_interface(q, weights, k, topk, offsets)
    #     # return ref_index_score(q, weights, k, topk, offsets)

    # ref_time = do_bench(fn, rep=10000, warmup=100)


if __name__ == '__main__':
    test_kernel()