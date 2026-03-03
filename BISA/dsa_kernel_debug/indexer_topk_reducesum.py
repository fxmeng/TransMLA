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
        
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            
            q_chunk = q_batch[chunk_start:chunk_end]
            weights_chunk = weights_batch[chunk_start:chunk_end]
            
            k_visible = k_batch[:chunk_end]
            
            logits = einsum(q_chunk, k_visible, 's1 h d, s2 d -> s1 h s2')
            logits = F.relu(logits)
            
            logits = (logits * weights_chunk.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
            
            row_indices = torch.arange(chunk_start, chunk_end, device=device)[:, None]
            col_indices = torch.arange(chunk_end, device=device)[None, :]
            mask = row_indices >= col_indices
            
            logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=device))
            
            if chunk_end < topk:
                pad_size = topk - chunk_end
                logits = F.pad(logits, (0, pad_size), value=float('-inf'))
            
            topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
            topk_scores = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
            
            if chunk_end < topk:
                valid_mask = topk_indices < chunk_end
                topk_indices = torch.where(valid_mask, topk_indices, torch.tensor(-1, dtype=torch.int64, device=device))
                topk_scores = torch.where(valid_mask, topk_scores, torch.tensor(float('-inf'), device=device))
            
            global_start = start_idx + chunk_start
            global_end = start_idx + chunk_end
            all_topk_indices[global_start:global_end] = topk_indices
            all_topk_score[global_start:global_end] = topk_scores
    
    return all_topk_indices, all_topk_score


def ref_index_score(Q: torch.Tensor, Weights: torch.Tensor, K: torch.Tensor, topk: int,
                    offsets: torch.Tensor):
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
    S=8192,
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
    offsets = torch.tensor([0, 1000, 2000, 3000, 8192], dtype=torch.int32).cuda()

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