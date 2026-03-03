import torch
import torch.nn.functional as F
from einops import einsum, repeat

import tilelang as tl
import tilelang.language as T
from typing import Optional
from index import prepare_token_indices

from util import get_abs_err, get_err_ratio

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

pass_configs = {
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tl.jit(pass_configs=pass_configs)
def tl_full_indexer_bwd_impl(
    heads: int,
    head_dim: int,
    index_dim: int,
    sm_scale: Optional[float] = None,
    block_K: int = 32,
    block_H: int = 32,  # Head blocking to reduce shared memory
    num_stages: int = 0,
    num_threads: int = 128,
):
    """
    Full indexer backward kernel with head blocking.
    Instead of loading full Q[heads, head_dim] into shared memory,
    we process heads in blocks of block_H to reduce memory usage.
    """
    assert num_stages == 0
    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")
    dtype: str = BF16
    accum_dtype: str = FP32
    
    q_shape = [seq_len, heads, head_dim]
    kv_shape = [seq_len, head_dim]
    index_q_shape = [seq_len, heads, index_dim]
    weights_shape = [seq_len, heads]
    index_k_shape = [seq_len, index_dim]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    
    dindex_q_shape = [seq_len, heads, index_dim]
    dweights_shape = [seq_len, heads]
    dindex_k_shape = [seq_len, index_dim]
    
    num_h_blocks = T.ceildiv(heads, block_H)
    
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    @T.prim_func
    def tl_full_indexer_bwd_kernel(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            IndexQ: T.Tensor(index_q_shape, dtype),
            Weights: T.Tensor(weights_shape, dtype),
            IndexK: T.Tensor(index_k_shape, dtype),
            dIndexQ: T.Tensor(dindex_q_shape, accum_dtype),
            dWeights: T.Tensor(dweights_shape, accum_dtype),
            dIndexK: T.Tensor(dindex_k_shape, accum_dtype),
            Offsets: T.Tensor(offsets_shape, INT32),
            TokenIndices: T.Tensor(token_indices_shape, INT32),
    ):
        with T.Kernel(seq_len, threads=num_threads) as (bx):
            i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
            bos = Offsets[i_b]
            cur_seq_len = i_t + 1
            num_k_blocks = T.ceildiv(cur_seq_len, block_K)

            # Small shared buffers - head blocking version
            k_shared = T.alloc_shared([block_K, head_dim], dtype=dtype)
            index_k_shared = T.alloc_shared([block_K, index_dim], dtype=dtype)
            
            # Per-head-block buffers (much smaller than [heads, head_dim])
            q_block = T.alloc_shared([block_H, head_dim], dtype=dtype)
            index_q_block = T.alloc_shared([block_H, index_dim], dtype=dtype)
            weights_block = T.alloc_shared([block_H], dtype=dtype)
            
            # Per-head LSE and total sum
            lse_attn_shared = T.alloc_shared([heads], dtype=accum_dtype)
            lse_idx_shared = T.alloc_shared([1], dtype=accum_dtype)
            
            # Initialize
            for h in T.Parallel(heads):
                lse_attn_shared[h] = float('-inf')
            lse_idx_shared[0] = float('-inf')

            # ============ First Pass: Compute LSE ============
            for bi_k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                k_st = bi_k * block_K
                
                # Load K block
                for i, j in T.Parallel(block_K, head_dim):
                    pos = k_st + i
                    k_shared[i, j] = T.if_then_else(pos < cur_seq_len, K[bos + pos, j], 0)
                
                # Load IndexK block
                for i, j in T.Parallel(block_K, index_dim):
                    pos = k_st + i
                    index_k_shared[i, j] = T.if_then_else(pos < cur_seq_len, IndexK[bos + pos, j], 0)

                # Initialize idx_score for this K block
                idx_score_block = T.alloc_shared([block_K], dtype=accum_dtype)
                for i in T.Parallel(block_K):
                    idx_score_block[i] = 0

                # Process heads in blocks
                for bi_h in T.serial(num_h_blocks):
                    h_st = bi_h * block_H
                    
                    # Load Q block for this head range
                    for i, j in T.Parallel(block_H, head_dim):
                        h_idx = h_st + i
                        if h_idx < heads:
                            q_block[i, j] = Q[bos + i_t, h_idx, j]
                        else:
                            q_block[i, j] = 0
                    
                    # Load IndexQ block (with sm_scale)
                    for i, j in T.Parallel(block_H, index_dim):
                        h_idx = h_st + i
                        if h_idx < heads:
                            index_q_block[i, j] = IndexQ[bos + i_t, h_idx, j] * sm_scale
                        else:
                            index_q_block[i, j] = 0
                    
                    # Load weights block
                    for i in T.Parallel(block_H):
                        h_idx = h_st + i
                        if h_idx < heads:
                            weights_block[i] = Weights[bos + i_t, h_idx]
                        else:
                            weights_block[i] = 0

                    # Compute attn logits: k_shared[block_K, head_dim] @ q_block.T[head_dim, block_H] -> [block_K, block_H]
                    attn_logits = T.alloc_fragment([block_K, block_H], accum_dtype)
                    T.gemm(k_shared, q_block, attn_logits, 
                           transpose_A=False, transpose_B=True, clear_accum=True)
                    
                    attn_logits_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i, j in T.Parallel(block_K, block_H):
                        val = attn_logits[i, j] * sm_scale
                        if k_st + i >= cur_seq_len:
                            val = float('-inf')
                        attn_logits_shared[i, j] = val

                    # Update per-head LSE
                    for j in T.serial(block_H):
                        h_idx = h_st + j
                        if h_idx < heads:
                            for i in T.serial(block_K):
                                old_lse = lse_attn_shared[h_idx]
                                new_val = attn_logits_shared[i, j]
                                max_val = T.max(old_lse, new_val)
                                lse_attn_shared[h_idx] = max_val + T.log(
                                    T.exp(old_lse - max_val) + T.exp(new_val - max_val))

                    # Compute idx logits: index_k_shared[block_K, index_dim] @ index_q_block.T -> [block_K, block_H]
                    idx_logits = T.alloc_fragment([block_K, block_H], accum_dtype)
                    T.gemm(index_k_shared, index_q_block, idx_logits,
                           transpose_A=False, transpose_B=True, clear_accum=True)
                    
                    idx_relu_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i, j in T.Parallel(block_K, block_H):
                        idx_relu_shared[i, j] = T.max(idx_logits[i, j], 0)

                    # Accumulate weighted sum for idx_score
                    for i in T.serial(block_K):
                        for j in T.serial(block_H):
                            h_idx = h_st + j
                            if h_idx < heads:
                                idx_score_block[i] = idx_score_block[i] + weights_block[j] * idx_relu_shared[i, j]

                # Update idx LSE after all heads processed
                for i in T.serial(block_K):
                    score = idx_score_block[i]
                    if k_st + i >= cur_seq_len:
                        score = float('-inf')
                    old_lse = lse_idx_shared[0]
                    max_val = T.max(old_lse, score)
                    lse_idx_shared[0] = max_val + T.log(
                        T.exp(old_lse - max_val) + T.exp(score - max_val))

            # ============ Second Pass: Compute attn_sum_total ============
            # Must use shared for cross-thread accumulation
            attn_sum_total = T.alloc_shared([1], dtype=accum_dtype)
            attn_sum_total[0] = 0
            
            for bi_k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                k_st = bi_k * block_K
                
                for i, j in T.Parallel(block_K, head_dim):
                    pos = k_st + i
                    k_shared[i, j] = T.if_then_else(pos < cur_seq_len, K[bos + pos, j], 0)

                for bi_h in T.serial(num_h_blocks):
                    h_st = bi_h * block_H
                    
                    for i, j in T.Parallel(block_H, head_dim):
                        h_idx = h_st + i
                        if h_idx < heads:
                            q_block[i, j] = Q[bos + i_t, h_idx, j]
                        else:
                            q_block[i, j] = 0

                    attn_logits = T.alloc_fragment([block_K, block_H], accum_dtype)
                    T.gemm(k_shared, q_block, attn_logits,
                           transpose_A=False, transpose_B=True, clear_accum=True)
                    
                    attn_probs_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i, j in T.Parallel(block_K, block_H):
                        val = attn_logits[i, j] * sm_scale
                        h_idx = h_st + j
                        if k_st + i >= cur_seq_len or h_idx >= heads:
                            val = float('-inf')
                        if h_idx < heads:
                            attn_probs_shared[i, j] = T.exp(val - lse_attn_shared[h_idx])
                        else:
                            attn_probs_shared[i, j] = 0

                    for i in T.serial(block_K):
                        if k_st + i < cur_seq_len:
                            for j in T.serial(block_H):
                                h_idx = h_st + j
                                if h_idx < heads:
                                    attn_sum_total[0] = attn_sum_total[0] + attn_probs_shared[i, j]

            # ============ Third Pass: Compute Gradients ============
            d_index_q_shared = T.alloc_shared([heads, index_dim], dtype=accum_dtype)
            d_weights_shared = T.alloc_shared([heads], dtype=accum_dtype)
            for i, j in T.Parallel(heads, index_dim):
                d_index_q_shared[i, j] = 0
            for h in T.Parallel(heads):
                d_weights_shared[h] = 0

            for bi_k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                k_st = bi_k * block_K
                
                # Load K and IndexK
                for i, j in T.Parallel(block_K, head_dim):
                    pos = k_st + i
                    k_shared[i, j] = T.if_then_else(pos < cur_seq_len, K[bos + pos, j], 0)
                
                for i, j in T.Parallel(block_K, index_dim):
                    pos = k_st + i
                    index_k_shared[i, j] = T.if_then_else(pos < cur_seq_len, IndexK[bos + pos, j], 0)

                # Recompute idx_score and attn_sum for this K block
                attn_sum_block = T.alloc_shared([block_K], dtype=accum_dtype)
                idx_score_block = T.alloc_shared([block_K], dtype=accum_dtype)
                for i in T.Parallel(block_K):
                    attn_sum_block[i] = 0
                    idx_score_block[i] = 0

                for bi_h in T.serial(num_h_blocks):
                    h_st = bi_h * block_H
                    
                    for i, j in T.Parallel(block_H, head_dim):
                        h_idx = h_st + i
                        if h_idx < heads:
                            q_block[i, j] = Q[bos + i_t, h_idx, j]
                        else:
                            q_block[i, j] = 0
                    
                    for i, j in T.Parallel(block_H, index_dim):
                        h_idx = h_st + i
                        if h_idx < heads:
                            index_q_block[i, j] = IndexQ[bos + i_t, h_idx, j] * sm_scale
                        else:
                            index_q_block[i, j] = 0
                    
                    for i in T.Parallel(block_H):
                        h_idx = h_st + i
                        if h_idx < heads:
                            weights_block[i] = Weights[bos + i_t, h_idx]
                        else:
                            weights_block[i] = 0

                    # Compute attn probs
                    attn_logits = T.alloc_fragment([block_K, block_H], accum_dtype)
                    T.gemm(k_shared, q_block, attn_logits,
                           transpose_A=False, transpose_B=True, clear_accum=True)
                    
                    attn_probs_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i, j in T.Parallel(block_K, block_H):
                        val = attn_logits[i, j] * sm_scale
                        h_idx = h_st + j
                        if k_st + i >= cur_seq_len or h_idx >= heads:
                            val = float('-inf')
                        if h_idx < heads:
                            attn_probs_shared[i, j] = T.exp(val - lse_attn_shared[h_idx])
                        else:
                            attn_probs_shared[i, j] = 0

                    for i in T.serial(block_K):
                        for j in T.serial(block_H):
                            h_idx = h_st + j
                            if h_idx < heads:
                                attn_sum_block[i] = attn_sum_block[i] + attn_probs_shared[i, j]

                    # Compute idx logits
                    idx_logits = T.alloc_fragment([block_K, block_H], accum_dtype)
                    T.gemm(index_k_shared, index_q_block, idx_logits,
                           transpose_A=False, transpose_B=True, clear_accum=True)
                    
                    idx_relu_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i, j in T.Parallel(block_K, block_H):
                        idx_relu_shared[i, j] = T.max(idx_logits[i, j], 0)

                    for i in T.serial(block_K):
                        for j in T.serial(block_H):
                            h_idx = h_st + j
                            if h_idx < heads:
                                idx_score_block[i] = idx_score_block[i] + weights_block[j] * idx_relu_shared[i, j]

                # Compute d_idx_score
                d_idx_score_shared = T.alloc_shared([block_K], dtype=accum_dtype)
                for i in T.serial(block_K):
                    idx_s = idx_score_block[i]
                    if k_st + i >= cur_seq_len:
                        idx_s = float('-inf')
                    idx_prob = T.exp(idx_s - lse_idx_shared[0])
                    attn_prob = attn_sum_block[i] / (attn_sum_total[0] + 1e-9)
                    d_idx_score_shared[i] = idx_prob - attn_prob

                # Compute gradients per head block
                d_index_k_shared = T.alloc_shared([block_K, index_dim], accum_dtype)
                for i, j in T.Parallel(block_K, index_dim):
                    d_index_k_shared[i, j] = 0

                for bi_h in T.serial(num_h_blocks):
                    h_st = bi_h * block_H
                    
                    # Reload index_q_block and weights_block
                    for i, j in T.Parallel(block_H, index_dim):
                        h_idx = h_st + i
                        if h_idx < heads:
                            index_q_block[i, j] = IndexQ[bos + i_t, h_idx, j] * sm_scale
                        else:
                            index_q_block[i, j] = 0
                    
                    for i in T.Parallel(block_H):
                        h_idx = h_st + i
                        if h_idx < heads:
                            weights_block[i] = Weights[bos + i_t, h_idx]
                        else:
                            weights_block[i] = 0

                    # Recompute idx logits
                    idx_logits = T.alloc_fragment([block_K, block_H], accum_dtype)
                    T.gemm(index_k_shared, index_q_block, idx_logits,
                           transpose_A=False, transpose_B=True, clear_accum=True)
                    
                    idx_logits_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    idx_relu_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i, j in T.Parallel(block_K, block_H):
                        idx_logits_shared[i, j] = idx_logits[i, j]
                        idx_relu_shared[i, j] = T.max(idx_logits[i, j], 0)

                    # d_logits = d_idx_score * weights * (logits > 0)
                    d_logits_shared = T.alloc_shared([block_K, block_H], accum_dtype)
                    for i in T.serial(block_K):
                        for j in T.serial(block_H):
                            d_relu = T.if_then_else(idx_logits_shared[i, j] > 0, 1.0, 0.0)
                            d_logits_shared[i, j] = d_idx_score_shared[i] * weights_block[j] * d_relu

                    # d_weights
                    for j in T.serial(block_H):
                        h_idx = h_st + j
                        if h_idx < heads:
                            for i in T.serial(block_K):
                                if k_st + i < cur_seq_len:
                                    d_weights_shared[h_idx] = d_weights_shared[h_idx] + d_idx_score_shared[i] * idx_relu_shared[i, j]

                    # d_IndexQ
                    for h in T.serial(block_H):
                        h_idx = h_st + h
                        if h_idx < heads:
                            for d in T.serial(index_dim):
                                acc = T.alloc_var(accum_dtype)
                                acc = 0
                                for k in T.serial(block_K):
                                    acc = acc + d_logits_shared[k, h] * index_k_shared[k, d]
                                d_index_q_shared[h_idx, d] = d_index_q_shared[h_idx, d] + acc * sm_scale

                    # d_IndexK
                    for i in T.serial(block_K):
                        for k in T.serial(index_dim):
                            acc = T.alloc_var(accum_dtype)
                            acc = 0
                            for j in T.serial(block_H):
                                h_idx = h_st + j
                                if h_idx < heads:
                                    # Use original IndexQ (not scaled) then multiply by sm_scale
                                    acc = acc + d_logits_shared[i, j] * IndexQ[bos + i_t, h_idx, k]
                            d_index_k_shared[i, k] = d_index_k_shared[i, k] + acc * sm_scale

                # Write d_IndexK
                for i, j in T.Parallel(block_K, index_dim):
                    pos = k_st + i
                    if pos < cur_seq_len:
                        T.atomic_add(dIndexK[bos + pos, j], d_index_k_shared[i, j])

            # Write outputs
            T.copy(d_index_q_shared, dIndexQ[bos + i_t, :, :])
            T.copy(d_weights_shared, dWeights[bos + i_t, :])

    return tl_full_indexer_bwd_kernel


def full_indexer_bwd_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    indexq: torch.Tensor,
    weights: torch.Tensor,
    indexk: torch.Tensor,
    offsets: torch.Tensor,
):
    seq_len, heads, head_dim = q.shape
    _, _, index_dim = indexq.shape
    token_indices = prepare_token_indices(offsets)
    # Use float32 for gradients
    dindexq = torch.zeros(seq_len, heads, index_dim, dtype=torch.float32, device=indexq.device)
    dweights = torch.zeros(seq_len, heads, dtype=torch.float32, device=weights.device)
    dindexk = torch.zeros(seq_len, index_dim, dtype=torch.float32, device=indexk.device)
    kernel = tl_full_indexer_bwd_impl(heads, head_dim, index_dim)
    kernel(q, k, indexq, weights, indexk, dindexq, dweights, dindexk, offsets, token_indices)
    # Convert back to input dtype
    return dindexq.to(indexq.dtype), dweights.to(weights.dtype), dindexk.to(indexk.dtype)


def ref_indexer_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    IndexQ: torch.Tensor, 
    Weights: torch.Tensor, 
    IndexK: torch.Tensor,
    offsets: torch.Tensor) -> torch.Tensor:

    IndexQ.requires_grad_(True)
    Weights.requires_grad_(True)
    IndexK.requires_grad_(True)
    IndexQ.grad = None
    Weights.grad = None
    IndexK.grad = None
    softmax_scale = Q.shape[-1]**-0.5
    all_loss = []
    all_log_topk_prob = []

    for i in range(offsets.shape[0] - 1):
        q = Q[offsets[i]:offsets[i + 1]]
        k = K[offsets[i]:offsets[i + 1]]
        s = offsets[i + 1] - offsets[i]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).unsqueeze(-2).to(q.device)
        attn_score = einsum(q, k, 'q h d, k d -> q h k') * softmax_scale
        attn_score = torch.where(mask, attn_score, float('-inf'))
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = attn_score.sum(dim=-2)
        attn_score = attn_score / (attn_score.sum(dim=-1, keepdim=True) + 1e-9)

        indexq = IndexQ[offsets[i]:offsets[i + 1]]
        weights = Weights[offsets[i]:offsets[i + 1]]
        indexk = IndexK[offsets[i]:offsets[i + 1]]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = einsum(indexq, indexk, 's1 h k, s2 k -> s1 h s2') * softmax_scale
        logits = F.relu(logits)
        index_score = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32)
        index_score = torch.where(mask, index_score, float('-inf'))
        index_score = F.log_softmax(index_score, dim=-1)

        loss = F.kl_div(
            index_score.clip(-100, 0),
            attn_score.log().clip(-100, 0),
            log_target=True,
            reduction="sum")
        all_loss.append(loss)

    loss = torch.stack(all_loss).sum()
    loss.backward()

    return IndexQ.grad, Weights.grad, IndexK.grad




def test_kernel(
    B=1,
    S=8192,
    H=128,
    D=576,
    d=128,
):
    torch.manual_seed(999)
    q = torch.randn((S, H, D)).cuda().bfloat16()
    kv = torch.randn((S, D)).cuda().bfloat16()
    indexq = torch.randn((S, H, d)).cuda().bfloat16()
    indexk = torch.randn((S, d)).cuda().bfloat16()
    weights = torch.randn((S, H)).cuda().bfloat16()
    offsets = torch.tensor([0, 4000, 32768], dtype=torch.int32).cuda()


    ref_dq, ref_dw, ref_dk = ref_indexer_bwd(q, kv, indexq, weights, indexk, offsets)

    dq, dw, dk = full_indexer_bwd_interface(q, kv, indexq, weights, indexk, offsets)

    print(f"dq err: {get_abs_err(dq, ref_dq):.6f} ratio: {get_err_ratio(dq, ref_dq):.6f}")
    print(f"dw err: {get_abs_err(dw, ref_dw):.6f} ratio: {get_err_ratio(dw, ref_dw):.6f}")
    print(f"dk err: {get_abs_err(dk, ref_dk):.6f} ratio: {get_err_ratio(dk, ref_dk):.6f}")


if __name__ == '__main__':
    test_kernel()