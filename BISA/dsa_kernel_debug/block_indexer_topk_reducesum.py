import math
import torch
import torch.nn.functional as F
from einops import einsum

import tilelang as tl
import tilelang.language as T
from typing import Optional

from util import get_abs_err, get_err_ratio

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_pass_configs)
def tl_gather_qk_reducesum_impl(
        heads,
        dim,
        num_candidates,
        sm_scale=None,
        block_I=32,
        num_stages=2,
        threads=128,
):
    """
    Fused kernel: gather K by indices + QK matmul + ReLU + weighted reduce-sum.

    Args:
        heads: number of attention heads H
        dim: head dimension D
        num_candidates: total number of candidate tokens per query (block_topk * block_size)
        sm_scale: softmax scale factor (default: dim ** -0.5)
        block_I: tile size for iterating over candidates
        num_stages: pipeline stages
        threads: threads per block
    """
    if sm_scale is None:
        sm_scale = dim ** -0.5

    assert num_candidates % block_I == 0, \
        f"num_candidates ({num_candidates}) must be divisible by block_I ({block_I})"

    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")
    batch_plus_one = T.symbolic("batch_plus_one")

    H = heads
    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    D = dim
    BI = block_I
    NI = num_candidates // block_I

    dtype = "bfloat16"
    accum_dtype = "float"
    indices_dtype = "int32"

    q_shape = [seq_len, H, D]
    k_shape = [seq_len_kv, D]
    weights_shape = [seq_len, H]
    token_indices_shape = [seq_len, num_candidates]
    offsets_shape = [batch_plus_one]
    seq_token_indices_shape = [seq_len, 2]
    score_shape = [seq_len, num_candidates]

    @T.prim_func
    def gather_qk_reducesum_kernel(
            Q: T.Tensor(q_shape, dtype),  # [seq_len, H, D]
            K: T.Tensor(k_shape, dtype),  # [seq_len_kv, D]
            Weights: T.Tensor(weights_shape, dtype),  # [seq_len, H]
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # [seq_len, num_candidates]
            Offsets: T.Tensor(offsets_shape, indices_dtype),  # [batch_plus_one]
            SeqTokenIndices: T.Tensor(seq_token_indices_shape, indices_dtype),  # [seq_len, 2]
            Score: T.Tensor(score_shape, accum_dtype),  # [seq_len, num_candidates]
    ):
        with T.Kernel(seq_len, threads=threads) as (bx,):
            # Shared memory allocations
            Q_shared = T.alloc_shared([padded_H, D], dtype)
            K_shared = T.alloc_shared([BI, D], dtype)
            W_shared = T.alloc_fragment([padded_H], accum_dtype)

            # Fragment for accumulating QK scores: [padded_H, BI]
            acc_s = T.alloc_fragment([padded_H, BI], accum_dtype)
            # Fragment for weighted reduced score per block: [BI]
            reducesum = T.alloc_fragment([BI], accum_dtype)

            # Resolve batch index and sequence position
            b_i, s_i = SeqTokenIndices[bx, 0], SeqTokenIndices[bx, 1]
            bos = Offsets[b_i]

            # Load Q[bos + s_i, :, :] into shared memory
            T.copy(Q[bos + s_i, 0:padded_H, :D], Q_shared)

            # Load Weights[bos + s_i, :] into fragment
            for h_i in T.Parallel(padded_H):
                if h_i < H:
                    W_shared[h_i] = Weights[bos + s_i, h_i]
                else:
                    W_shared[h_i] = 0

            # Iterate over candidate blocks
            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # Gather K: load K[token_indices[bos+s_i, i_i*BI + bi_i]] into K_shared
                # token_indices == -1 means invalid, clamp to 0 for safe load
                for bi_i, d_i in T.Parallel(BI, D):
                    K_shared[bi_i, d_i] = K[
                        bos + T.max(TokenIndices[bos + s_i, i_i * BI + bi_i], 0),
                        d_i
                    ]

                # Initialize acc_s: set to 0 for valid indices, -inf for invalid (token_indices == -1)
                for h_i, bi_i in T.Parallel(padded_H, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        TokenIndices[bos + s_i, i_i * BI + bi_i] >= 0,
                        0,
                        -T.infinity(acc_s.dtype)
                    )

                # QK matmul: acc_s += Q_shared @ K_shared^T  -> [padded_H, BI]
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # ReLU + scale + weighted reduce-sum over heads
                for h_i, bi_i in T.Parallel(padded_H, BI):
                    # ReLU
                    acc_s[h_i, bi_i] = T.max(acc_s[h_i, bi_i], 0)
                    # Multiply by weight and scale
                    acc_s[h_i, bi_i] = acc_s[h_i, bi_i] * W_shared[h_i] * sm_scale

                # Reduce sum over heads: reducesum[bi] = sum_h acc_s[h, bi]
                T.reduce_sum(acc_s, reducesum, dim=0)

                # Store result
                T.copy(reducesum, Score[bos + s_i, i_i * BI:i_i * BI + BI])

    return gather_qk_reducesum_kernel


def gather_qk_reducesum_interface(
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        token_indices: torch.Tensor,
        offsets: torch.Tensor,
        sm_scale: float = None,
        block_I: int = 32,
        num_stages: int = 2,
        threads: int = 128,
) -> torch.Tensor:
    """
    Fused gather-K + QK-matmul + ReLU + weighted-reducesum interface.

    Args:
        q: [total_seq_len, H, D] bf16 query tensor
        k: [total_seq_len, D] bf16 key tensor
        weights: [total_seq_len, H] bf16 weight tensor
        token_indices: [total_seq_len, num_candidates] int32 candidate token indices (-1 for invalid)
        offsets: [batch+1] int32 batch offsets
        sm_scale: softmax scale (default: D ** -0.5)
        block_I: tile size for candidates dimension
        num_stages: pipeline stages
        threads: threads per block

    Returns:
        score: [total_seq_len, num_candidates] float32 weighted scores
    """
    seq_len, H, D = q.shape
    num_candidates = token_indices.shape[1]

    if sm_scale is None:
        sm_scale = D ** -0.5

    # Prepare sequence/token indices for the kernel
    seq_token_indices = prepare_token_indices(offsets)

    # Ensure token_indices is int32
    if token_indices.dtype != torch.int32:
        token_indices = token_indices.to(torch.int32)

    # Allocate output
    score = torch.zeros(seq_len, num_candidates, dtype=torch.float32, device=q.device)

    # Get compiled kernel
    kernel = tl_gather_qk_reducesum_impl(
        heads=H,
        dim=D,
        num_candidates=num_candidates,
        sm_scale=sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )

    # Run kernel
    kernel(q, k, weights, token_indices, offsets, seq_token_indices, score)

    return score


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

            q_chunk = q_batch[chunk_start:chunk_end]  # [chunk_len, H, D]
            weights_chunk = weights_batch[chunk_start:chunk_end]  # [chunk_len, H]
            chunk_end_block = math.ceil(chunk_end / block_size)
            k_block_mean_visible = k_block_mean[:chunk_end_block]  # [chunk_end_block, D]           

            # Step 1: Compute block-level scores [chunk_len, chunk_end_block]
            block_logits = einsum(q_chunk, k_block_mean_visible, 'cl h d, nb d -> cl h nb')
            block_logits = F.relu(block_logits)
            block_scores = (block_logits * weights_chunk.unsqueeze(-1)).sum(dim=-2,
                                                                            dtype=torch.float32) * softmax_scale  # [chunk_len, chunk_end_block]

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
            _, selected_block_indices = torch.topk(block_scores_for_topk, k=actual_btopk,
                                                   dim=-1)  # [chunk_len, actual_btopk]

            # Step 3: Expand selected blocks into token indices [chunk_len, block_topk * block_size]
            block_start_tokens = selected_block_indices * block_size  # [chunk_len, actual_btopk]
            offsets_in_block = torch.arange(block_size, device=device)  # [block_size]
            candidate_indices = (block_start_tokens.unsqueeze(-1) + offsets_in_block).reshape(chunk_len,
                                                                                              -1)  # [chunk_len, actual_btopk * block_size]

            valid_mask = (candidate_indices <= q_positions.unsqueeze(1)) & (candidate_indices < seq_len)
            token_indices = torch.where(valid_mask, candidate_indices, torch.tensor(-1, dtype=torch.int64,
                                                                                    device=device))  # [chunk_len, actual_btopk * block_size]

            # Pad to full width if actual_btopk < block_topk
            if token_indices.shape[1] < block_topk * block_size:
                token_indices = F.pad(token_indices, (0, block_topk * block_size - token_indices.shape[1]), value=-1)

            # Step 4: Compute token-level scores for selected tokens (fused gather + QK + ReLU + reducesum)
            num_candidates = token_indices.shape[1]  # block_topk * block_size
            # Build per-chunk offsets for the fused kernel (single chunk = single "batch")
            chunk_offsets = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
            token_scores = gather_qk_reducesum_interface(
                q_chunk, k_batch, weights_chunk,
                token_indices.to(torch.int32),
                chunk_offsets,
                sm_scale=softmax_scale,
                block_I=32,
                num_stages=2,
                threads=128,
            )  # [chunk_len, num_candidates]
            token_scores = torch.where(token_indices >= 0, token_scores, torch.tensor(float('-inf'), device=device))

            # Step 5: Topk token selection and store into all_topk_indices/all_topk_score
            actual_topk = min(topk, num_candidates)
            topk_scores, topk_local_ids = torch.topk(token_scores, k=actual_topk, dim=-1)  # [chunk_len, actual_topk]
            topk_token_indices = torch.gather(token_indices, dim=1, index=topk_local_ids)  # [chunk_len, actual_topk]

            # Keep batch-local indices (consistent with ref_index_score)
            topk_global_indices = torch.where(topk_token_indices >= 0, topk_token_indices,
                                              torch.tensor(-1, dtype=torch.int64, device=device))
            topk_final_scores = F.softmax(topk_scores, dim=-1, dtype=torch.float32)
            topk_final_scores = torch.where(topk_token_indices >= 0, topk_final_scores,
                                            torch.tensor(float('-inf'), device=device))

            # Store results
            global_start = start_idx + chunk_start
            global_end = start_idx + chunk_end
            if actual_topk < topk:
                all_topk_indices[global_start:global_end, :actual_topk] = topk_global_indices
                all_topk_score[global_start:global_end, :actual_topk] = topk_final_scores
            else:
                all_topk_indices[global_start:global_end] = topk_global_indices
                all_topk_score[global_start:global_end] = topk_final_scores

    all_topk_indices = all_topk_indices.to(torch.int32)

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

        if s <= topk:
            # Case 1: s <= topk, directly generate topk_indices and topk_score without any selection
            mask = (torch.arange(s, device=q.device)[:, None] >= torch.arange(s, device=q.device)[None, :])
            logits = einsum(q, k, 's1 h d, s2 d -> s1 h s2')
            logits = F.relu(logits)
            logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
            logits = torch.where(mask, logits, float('-inf'))

            # Pad to topk since s < topk
            pad_size = topk - s
            logits = F.pad(logits, (0, pad_size), value=float('-inf'))

            topk_logits, topk_indices_batch = torch.topk(logits, k=topk, dim=-1)
            topk_score_batch = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

            valid_mask = topk_indices_batch < s
            topk_score_batch = torch.where(valid_mask, topk_score_batch, torch.tensor(float('-inf'), device=q.device))
            topk_indices_batch = torch.where(valid_mask, topk_indices_batch, torch.tensor(-1, dtype=torch.int64, device=q.device))

            all_topk_indices.append(topk_indices_batch)
            all_topk_score.append(topk_score_batch)

        elif s <= block_size * block_topk:
            # Case 2: topk < s <= block_size * block_topk, token-level topk only (no block selection needed)
            mask = (torch.arange(s, device=q.device)[:, None] >= torch.arange(s, device=q.device)[None, :])
            logits = einsum(q, k, 's1 h d, s2 d -> s1 h s2')
            logits = F.relu(logits)
            logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
            logits = torch.where(mask, logits, float('-inf'))

            topk_logits, topk_indices_batch = torch.topk(logits, k=topk, dim=-1)
            topk_score_batch = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
            topk_score_batch = torch.where(topk_indices_batch < s, topk_score_batch, torch.tensor(float('-inf'), device=q.device))

            # Keep batch-local indices (no global offset)
            topk_indices_batch = torch.where(topk_indices_batch < s, topk_indices_batch, torch.tensor(-1, dtype=torch.int64, device=q.device))

            all_topk_indices.append(topk_indices_batch)
            all_topk_score.append(topk_score_batch)

        else:
            # Case 3: s > block_size * block_topk, two-stage indexer (block selection + token selection)

            # Step 1: Split keys into blocks and mean pool each block (pad + reshape + sum / block_counts)
            num_blocks = math.ceil(s / block_size)
            block_ranges = []
            for b in range(num_blocks):
                b_start = b * block_size
                b_end = min((b + 1) * block_size, s)
                block_ranges.append((b_start, b_end))

            pad_len = num_blocks * block_size - s
            if pad_len > 0:
                k_padded = F.pad(k, (0, 0, 0, pad_len), value=0.0)
                block_counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=q.device)
                block_counts[-1] = block_size - pad_len
            else:
                k_padded = k
                block_counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=q.device)
            k_block = (
                k_padded.reshape(num_blocks, block_size, -1)
                .sum(dim=1)
                / block_counts.unsqueeze(-1)
            ).to(k.dtype)  # [num_blocks, D]

            # Step 1: Per-token block selection -> generate block mask [s, s]
            block_mask = torch.zeros(s, s, dtype=torch.bool, device=q.device)

            LARGE_SCORE = 1e9
            for t in range(s):
                q_t = q[t]  # [H, D]
                w_t = weights[t]  # [H]

                # Only consider causal blocks (blocks whose start <= t)
                causal_block_ids = [b for b in range(num_blocks) if block_ranges[b][0] <= t]
                num_causal = len(causal_block_ids)

                if num_causal == 0:
                    continue

                # Compute block-level scores for all causal blocks
                causal_k = k_block[causal_block_ids]  # [num_causal, D]
                block_logits = einsum(q_t, causal_k, 'h d, n d -> h n')
                block_logits = F.relu(block_logits)
                block_scores = (block_logits * w_t.unsqueeze(-1)).sum(dim=0, dtype=torch.float32) * softmax_scale

                # Mandatory blocks: block 0, q_block_id, q_block_id - 1 (per-query causal)
                q_block_id = t // block_size
                mandatory_last1 = q_block_id
                mandatory_last2 = max(q_block_id - 1, 0)
                mandatory_set = {0, mandatory_last1, mandatory_last2}

                # Force mandatory blocks with LARGE_SCORE
                for mb in mandatory_set:
                    if mb in causal_block_ids:
                        local_idx = causal_block_ids.index(mb)
                        block_scores[local_idx] = LARGE_SCORE

                actual_btopk = min(block_topk, num_causal)
                _, top_block_local_ids = torch.topk(block_scores, k=actual_btopk, dim=-1)
                selected_block_ids = [causal_block_ids[idx] for idx in top_block_local_ids.cpu().tolist()]

                # Set block mask: mark selected block tokens as visible (respecting causality)
                for b in selected_block_ids:
                    b_start, b_end = block_ranges[b]
                    actual_end = min(b_end, t + 1)
                    if b_start < actual_end:
                        block_mask[t, b_start:actual_end] = True

            # Step 2: Compute token-level scores within selected blocks
            logits = einsum(q, k, 's1 h d, s2 d -> s1 h s2')
            logits = F.relu(logits)
            logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale

            # Apply causal mask
            causal_mask = (torch.arange(s, device=q.device)[:, None] >= torch.arange(s, device=q.device)[None, :])
            # Combine causal mask with block mask
            combined_mask = causal_mask & block_mask
            logits = torch.where(combined_mask, logits, torch.tensor(float('-inf'), device=q.device))

            # Token-level topk
            topk_logits, topk_indices_batch = torch.topk(logits, k=topk, dim=-1)
            topk_score_batch = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
            topk_score_batch = torch.where(topk_logits > float('-inf'), topk_score_batch, torch.tensor(float('-inf'), device=q.device))

            # Keep batch-local indices (no global offset)
            topk_indices_batch = torch.where(topk_logits > float('-inf'), topk_indices_batch, torch.tensor(-1, dtype=torch.int64, device=q.device))

            all_topk_indices.append(topk_indices_batch)
            all_topk_score.append(topk_score_batch)

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