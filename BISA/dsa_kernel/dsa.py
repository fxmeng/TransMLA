from typing import Optional
import torch
import torch.nn.functional as F
from .indexer_topk_reducesum import indexer_topk_reducesum_interface
from .block_indexer_topk_reducesum import indexer_topk_reducesum_interface as block_indexer_topk_reducesum_interface
from .indexer_bwd import indexer_bwd_interface
from .full_indexer_bwd import full_indexer_bwd_interface
from .sparse_mla_fwd import sparse_mla_fwd_interface
from .sparse_mla_bwd import sparse_mla_bwd
from .sparse_mla_topk_reducesum import sparse_mla_topk_reducesum_interface
from .dense_mla_fwd import dense_mla_fwd_interface
from einops import einsum, repeat

class DSAFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
        dim_v: int,
        sm_scale: Optional[float] = None,
    ):
        # topk_indices, index_score = ref_index_score(index_q, weights, index_k, topk)
        topk_indices, index_score = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
        o, lse = sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)
        ctx.save_for_backward(q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets)
        ctx.topk = topk
        ctx.dim_v = dim_v
        ctx.sm_scale = sm_scale
        return o, topk_indices

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
        _1: torch.Tensor,
    ):
        q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets = ctx.saved_tensors
        attn_score = sparse_mla_topk_reducesum_interface(
            q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), lse, offsets,
            dim_v=ctx.dim_v).squeeze(-2)
        dq, dkv = sparse_mla_bwd(
            q,
            kv.unsqueeze(-2),
            o,
            do,
            topk_indices.unsqueeze(-2),
            lse,
            offsets,
            sm_scale=ctx.sm_scale)
        dindex_q, dweights, dindex_k = indexer_bwd_interface(index_q, weights, index_k, attn_score,
                                                             index_score, topk_indices, offsets)
        return dq, dkv.squeeze(-2), dindex_q, dindex_k, dweights, None, None, None, None
        # return dq, dkv.squeeze(-2), None, None, None, None, None, None, None


def deepseek_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
):
    return DSAFunction.apply(q, kv, index_q, index_k, weights, offsets, topk, dim_v, sm_scale)

class DSAFunctionBlockIndexer(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
        dim_v: int,
        sm_scale: Optional[float] = None,
    ):
        # topk_indices, index_score = ref_index_score(index_q, weights, index_k, topk)
        topk_indices, index_score = block_indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
        o, lse = sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)
        ctx.save_for_backward(q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets)
        ctx.topk = topk
        ctx.dim_v = dim_v
        ctx.sm_scale = sm_scale
        return o, topk_indices

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
        _1: torch.Tensor,
    ):
        q, kv, index_q, index_k, weights, topk_indices, index_score, o, lse, offsets = ctx.saved_tensors
        attn_score = sparse_mla_topk_reducesum_interface(
            q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), lse, offsets,
            dim_v=ctx.dim_v).squeeze(-2)
        dq, dkv = sparse_mla_bwd(
            q,
            kv.unsqueeze(-2),
            o,
            do,
            topk_indices.unsqueeze(-2),
            lse,
            offsets,
            sm_scale=ctx.sm_scale)
        dindex_q, dweights, dindex_k = indexer_bwd_interface(index_q, weights, index_k, attn_score,
                                                             index_score, topk_indices, offsets)
        return dq, dkv.squeeze(-2), dindex_q, dindex_k, dweights, None, None, None, None
        # return dq, dkv.squeeze(-2), None, None, None, None, None, None, None


def deepseek_sparse_attention_block_indexer(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
):
    return DSAFunctionBlockIndexer.apply(q, kv, index_q, index_k, weights, offsets, topk, dim_v, sm_scale)


class DSAFunctionWarmup(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
        dim_v: int,
        sm_scale: Optional[float] = None,
    ):
        o, lse = dense_mla_fwd_interface(q, kv.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)
        ctx.save_for_backward(q, kv, index_q, index_k, weights, offsets)
        return o

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor
    ):
        q, kv, index_q, index_k, weights, offsets = ctx.saved_tensors

        dindex_q, dweights, dindex_k = full_indexer_bwd_interface(q, kv, index_q, weights, index_k, offsets)

        return None, None, dindex_q, dindex_k, dweights, None, None, None, None


def deepseek_sparse_attention_warmup(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float] = None,
):
    return DSAFunctionWarmup.apply(q, kv, index_q, index_k, weights, offsets, topk, dim_v, sm_scale)
