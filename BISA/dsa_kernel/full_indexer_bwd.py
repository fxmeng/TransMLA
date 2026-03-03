import torch
import torch.nn.functional as F
from einops import einsum, repeat

import tilelang as tl
import tilelang.language as T
from typing import Optional


@torch.no_grad()
def full_indexer_bwd_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    indexq: torch.Tensor,
    weights: torch.Tensor,
    indexk: torch.Tensor,
    offsets: torch.Tensor,
    chunk_size: int = 2048,
    eps: float = 1e-9,
):
    device = q.device
    softmax_scale = q.shape[-1] ** -0.5
    S, H, _ = q.shape
    d = indexq.shape[-1]

    dindexq = torch.zeros_like(indexq)
    dweights = torch.zeros_like(weights)
    dindexk = torch.zeros_like(indexk)

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    B = offsets.numel() - 1

    for bi in range(B):
        start = int(offsets[bi].item())
        end = int(offsets[bi + 1].item())
        seq_len = end - start
        if seq_len <= 0:
            continue

        q_batch = q[start:end]
        k_batch = k[start:end]
        indexq_batch = indexq[start:end]
        weights_batch = weights[start:end]
        indexk_batch = indexk[start:end]

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)

            q_chunk = q_batch[chunk_start:chunk_end]
            k_full = k_batch[:chunk_end]
            IQ = indexq_batch[chunk_start:chunk_end]
            W = weights_batch[chunk_start:chunk_end]
            IK = indexk_batch[:chunk_end]

            s1 = chunk_end - chunk_start
            s2 = chunk_end

            qp = torch.arange(chunk_start, chunk_end, device=device)[:, None]
            kp = torch.arange(chunk_end, device=device)[None, :]
            causal_2d = (qp >= kp)

            attn_logits = einsum(q_chunk, k_full, 'q h d, k d -> q h k') * softmax_scale
            attn_logits = attn_logits.masked_fill(~causal_2d.unsqueeze(1), float('-inf'))
            attn_prob_h = torch.softmax(attn_logits, dim=-1)

            p = attn_prob_h.sum(dim=1)
            p = p / (p.sum(dim=-1, keepdim=True) + eps)

            logp_clip = (p.clamp_min(eps).log()).clamp(-100.0, 0.0)
            p_used = logp_clip.exp().to(torch.float32)

            T = einsum(IQ, IK, 'i h k, j k -> i h j') * softmax_scale
            relu_mask = (T > 0)
            R = torch.relu(T)
        

            Sij = (R * W.unsqueeze(-1)).sum(dim=1).to(torch.float32)
            U = Sij.masked_fill(~causal_2d, float('-inf'))
            logq = torch.log_softmax(U, dim=-1)
            qhat = logq.exp()

            in_range = (logq > -100.0).to(torch.float32)
            logq_clip = torch.maximum(
                logq,
                torch.tensor(-100.0, device=device, dtype=logq.dtype),
            )

            loss = (p_used * (logp_clip.to(torch.float32) - logq_clip.to(torch.float32))).sum()
            total_loss += loss

            g = (-p_used * in_range).masked_fill(~causal_2d, 0.0)
            g_sum = g.sum(dim=-1, keepdim=True)
            dU = g - qhat.to(torch.float32) * g_sum
            dSij = dU

            dW = (dSij.unsqueeze(1) * R.to(torch.float32)).sum(dim=-1)
            dR = dSij.unsqueeze(1) * W.to(torch.float32).unsqueeze(-1)
            dT = dR * relu_mask.to(torch.float32)

            dIQ = softmax_scale * einsum(dT, IK.to(torch.float32), 'i h j, j k -> i h k')
            dIK = softmax_scale * einsum(dT, IQ.to(torch.float32), 'i h j, i h k -> j k')

            dindexq[start + chunk_start:start + chunk_end] += dIQ.to(dindexq.dtype)
            dweights[start + chunk_start:start + chunk_end] += dW.to(dweights.dtype)
            dindexk[start:start + chunk_end] += dIK.to(dindexk.dtype)
    
    return dindexq, dweights, dindexk
    
