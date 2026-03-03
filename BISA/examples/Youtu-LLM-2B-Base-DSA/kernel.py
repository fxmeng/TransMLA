from typing import Optional
import math
import torch
import torch.nn.functional as F
import tilelang
import tilelang as tl
import tilelang.language as T
import functools
from typing import Callable, Any

###################################################################
######################   prepare_cu_seqlens  ######################
###################################################################

def tensor_cache(fn: Callable[..., torch.Tensor],) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if (last_args is not None and last_kwargs is not None) and \
            (len(args) == len(last_args) and len(kwargs) == len(last_kwargs)) and \
                all(a is b for a, b in zip(args, last_args, strict=False)) and \
                    all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
            return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))

@tensor_cache
def cu_seqlens_from_position_ids(
    position_ids: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    starts = (position_ids == 0).nonzero(as_tuple=True)[0]
    total_len = position_ids.new_tensor([position_ids.numel()])
    boundaries = torch.cat([starts, total_len])
    lens = torch.diff(boundaries)
    cu_seqlens = prepare_cu_seqlens_from_lens(lens, dtype=dtype)
    return cu_seqlens

@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)

###################################################################
########################   Indexer Topk  ##########################
###################################################################

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

@tl.jit(
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def tl_indexer_topk_reducesum_impl(
    heads: int,
    dim: int,
    topk: int,
    sm_scale: Optional[float] = None,
    block_K: int = 32,
    dtype: str = FP32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert topk == tl.math.next_power_of_2(topk)
    assert topk % block_K == 0
    assert heads <= 64 and heads % 8 == 0
    assert num_stages == 0
    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    index_q_shape = [seq_len, heads, dim]
    weights_shape = [seq_len, heads]
    index_k_shape = [seq_len, dim]
    topk_indices_shape = [seq_len, topk]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    N = 2 * topk
    num_iters = int(round(math.log2(N)))
    if sm_scale is None:
        sm_scale = dim**-0.5

    @T.macro
    def bitonic_sort(
            topk_index_shared: T.SharedBuffer([N], dtype=INT32),
            topk_value_shared: T.SharedBuffer([N], dtype=FP32),
    ):
        T.sync_threads()
        for i1 in T.serial(num_iters):
            for i2 in T.serial(i1 + 1):
                for i in T.Parallel(N):
                    ascending = (i & (1 << (i1 + 1))) != 0
                    j = i ^ (1 << (i1 - i2))
                    if i < j and \
                        ((ascending and topk_value_shared[i] > topk_value_shared[j]) or (
                                not ascending and topk_value_shared[i] < topk_value_shared[j])):
                        val = topk_value_shared[i]
                        topk_value_shared[i] = topk_value_shared[j]
                        topk_value_shared[j] = val
                        idx = topk_index_shared[i]
                        topk_index_shared[i] = topk_index_shared[j]
                        topk_index_shared[j] = idx
                T.sync_threads()

    @T.prim_func
    def tl_indexer_topk_reducesum_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),
            Weights: T.Tensor(weights_shape, dtype),
            IndexK: T.Tensor(index_k_shape, dtype),
            TopkIndices: T.Tensor(topk_indices_shape, INT32),
            ReduceSum: T.Tensor(topk_indices_shape, FP32),
            Offsets: T.Tensor(offsets_shape, INT32),
            TokenIndices: T.Tensor(token_indices_shape, INT32),
    ):
        with T.Kernel(seq_len, threads=num_threads) as (bx):
            i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
            bos, eos = Offsets[i_b], Offsets[i_b + 1]
            num_blocks = T.ceildiv(i_t + 1, block_K)

            topk_index_shared = T.alloc_shared([N], dtype=INT32)
            topk_value_shared = T.alloc_shared([N], dtype=FP32)

            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float('-inf'))
            T.sync_threads()

            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            T.copy(IndexQ[bos + i_t, :, :], index_q_shared)
            T.sync_threads()

            weights_frag = T.alloc_shared([heads], dtype=dtype)
            T.copy(Weights[bos + i_t, :], weights_frag)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_st = bk_i * block_K
                k_ed = T.min((bk_i + 1) * block_K, eos - bos)

                index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                for i, j in T.Parallel(block_K, dim):
                    index_k_shared[i, j] = T.if_then_else(k_st + i < k_ed, IndexK[bos + k_st + i,
                                                                                  j], 0)
                T.sync_threads()

                logits = T.alloc_fragment((block_K, heads), FP32)
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )
                T.sync_threads()

                for i, j in T.Parallel(block_K, heads):
                    logits[i, j] = T.max(logits[i, j], 0) * weights_frag[j]
                T.sync_threads()

                logits_sum = T.alloc_fragment(block_K, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                T.sync_threads()

                offset = T.alloc_var(INT32)
                if k_st >= topk:
                    offset = topk + (k_st % topk)
                else:
                    offset = k_st
                T.sync_threads()
                for i in T.Parallel(block_K):
                    if k_st + i > i_t:
                        logits_sum[i] = float('-inf')
                    j = offset + i
                    topk_index_shared[j] = k_st + i
                    topk_value_shared[j] = logits_sum[i]
                T.sync_threads()

                if k_ed > topk and k_ed % topk == 0:
                    bitonic_sort(topk_index_shared, topk_value_shared)

            bitonic_sort(topk_index_shared, topk_value_shared)

            logits_max_frag = T.alloc_fragment([1], dtype=FP32)
            logits_frag = T.alloc_fragment([topk], dtype=FP32)
            reducesum_shared = T.alloc_shared([topk], dtype=FP32)

            T.copy(topk_value_shared[:topk], logits_frag)
            T.sync_threads()

            T.reduce_max(logits_frag, logits_max_frag, dim=-1)
            T.sync_threads()

            for i in T.Parallel(topk):
                logits_frag[i] = T.exp(logits_frag[i] - logits_max_frag[0])
            T.sync_threads()

            lse_frag = T.alloc_fragment([1], dtype=FP32)
            T.reduce_sum(logits_frag, lse_frag)
            T.sync_threads()

            for i in T.Parallel(topk):
                reducesum_shared[i] = logits_frag[i] / lse_frag[0]
            T.sync_threads()

            # for i in T.Parallel(topk):
            #     reducesum_shared[i] = logits_frag[i]
            # T.sync_threads()

            for i in T.Parallel(topk):
                if topk_index_shared[i] > i_t:
                    topk_index_shared[i] = -1
            T.sync_threads()

            T.copy(topk_index_shared[:topk], TopkIndices[bos + i_t, :])
            T.copy(reducesum_shared[:topk], ReduceSum[bos + i_t, :])

    return tl_indexer_topk_reducesum_kernel


def indexer_topk_reducesum_interface(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    offsets: torch.Tensor,
    dtype: str = BF16,
):
    seq_len, heads, dim = q.shape
    kernel = tl_indexer_topk_reducesum_impl(heads=heads, dim=dim, topk=topk, dtype=dtype)
    token_indices = prepare_token_indices(offsets)
    topk_indices = torch.zeros((seq_len, topk), device=q.device, dtype=torch.int32)
    topk_score = torch.zeros((seq_len, topk), device=q.device, dtype=torch.float32)
    kernel(q, weights, k, topk_indices, topk_score, offsets, token_indices)
    return topk_indices, topk_score

###################################################################
############################   DSA FWD  ###########################
###################################################################

@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=32,
    num_stages=2,
    threads=128,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (topk %
            block_I == 0), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5
    else:
        sm_scale = sm_scale

    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    head_kv = heads // kv_group
    q_shape = [seq_len, heads, dim + tail_dim]
    kv_shape = [seq_len, kv_group, dim + tail_dim]
    o_shape = [seq_len, heads, dim]
    indices_shape = [seq_len, kv_group, topk]
    lse_shape = [seq_len, heads]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert (
            kv_group == 1
        ), "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Offsets: T.Tensor(offsets_shape, indices_dtype),  # type: ignore
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
                seq_len * REPLICATE_H, kv_group, threads=threads) as (
                    bx,
                    by,
                ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]
            g_i = by
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
            T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i) & (
                        Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1)

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i,
                                              d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, g_i, i_i * BI + bi_i],
                                                  g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[bos + s_i, H0:H1, :])
            T.copy(sumexp, Lse[bos + s_i, H0:H1])

    return main


def sparse_mla_fwd_interface(q,
                             kv,
                             indices,
                             offsets,
                             sm_scale=None,
                             return_p_sum: bool = False,
                             d_v=512,
                             block_I=32,
                             num_stages=2,
                             threads=128):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    assert seq_len == seq_len_kv

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    _, _, topk = indices.shape
    assert indices.shape == (seq_len, kv_group, topk)

    token_indices = prepare_token_indices(offsets)

    kernel = sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads)
    out, lse = kernel(q, kv, indices, offsets, token_indices)
    return out, lse



###################################################################
############################   DSA BWD  ###########################
###################################################################

@tilelang.jit(out_idx=[-1])
def preprocess(
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype="bfloat16",
    accum_dtype="float",
):
    assert dtype == "bfloat16"
    assert accum_dtype == "float"

    S = T.symbolic('S')

    shape = [S, H, D]

    @T.prim_func
    def preprocess_kernel(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND)) as (bx, by):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[by * block_ND:(by + 1) * block_ND, bx, k * block_ND:(k + 1) * block_ND], o)
                T.copy(dO[by * block_ND:(by + 1) * block_ND, bx, k * block_ND:(k + 1) * block_ND],
                       do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[by * block_ND:(by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    D,
    D_tail,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype="bfloat16",
    accum_dtype="float",
):
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    S_kv = T.symbolic('S_kv')

    dkv_shape = [S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
            dKV: T.Tensor(dkv_shape, accum_dtype),
            dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, threads=threads) as (bx, by):
            T.copy(
                dKV[bx * block_N:(bx + 1) * block_N, by, :],
                dKV_out[bx * block_N:(bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def bwd(
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype="int32",
    dtype="bfloat16",
    accum_dtype="float",
):
    assert is_causal == True, 'non-casual is not supported now'
    assert topk % block_size == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'
    assert dtype == "bfloat16"
    assert accum_dtype == "float"
    assert indices_dtype == "int32"

    if sm_scale is None:
        sm_scale = (D + D_tail)**(-0.5)

    B_plus_one = T.symbolic('B_plus_one')
    S = T.symbolic('S')

    H_kv = H // kv_group
    q_shape = [S, H, D + D_tail]
    k_shape = [S, kv_group, D + D_tail]
    o_shape = [S, H, D]
    indices_shape = [S, kv_group, topk]
    delta_shape = [S, H]
    lse_shape = [S, H]
    offsets_shape = [B_plus_one]
    token_indices_shape = [S, 2]
    assert indices_dtype == "int32"
    assert dtype == "bfloat16"
    assert accum_dtype == "float"

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(k_shape, dtype),
            dO: T.Tensor(o_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            Lse: T.Tensor(lse_shape, accum_dtype),
            Delta: T.Tensor(delta_shape, accum_dtype),
            Offsets: T.Tensor(offsets_shape, indices_dtype),
            TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
            dQ: T.Tensor(q_shape, dtype),
            dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, kv_group, threads=threads) as (b_s_i, bz):
            Q_shared = T.alloc_shared([padded_H, D], dtype)
            Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([padded_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dQ_shared = T.alloc_shared([padded_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)

            acc_p = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([padded_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([padded_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            acc_dkv_shared = T.view(KV_shared, shape=[BS // split_store, D], dtype=accum_dtype)
            acc_dkv_tail_shared = T.view(
                KV_tail_shared, shape=[BS // split_store, D_tail], dtype=accum_dtype)

            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]

            max_kv_i = s_i

            T.copy(Q[bos + s_i, bz * padded_H:(bz + 1) * padded_H, :D], Q_shared)
            T.copy(Q[bos + s_i, bz * padded_H:(bz + 1) * padded_H, D:], Q_tail_shared)
            T.copy(dO[bos + s_i, bz * padded_H:(bz + 1) * padded_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            T.annotate_layout({
                dQ_shared: tilelang.layout.make_swizzled_layout(dQ_shared),
                dQ_tail_shared: tilelang.layout.make_swizzled_layout(dQ_tail_shared),
            })

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = (Indices[bos + s_i, bz, i_i * BS + bi_i] <= max_kv_i) & (
                        Indices[bos + s_i, bz, i_i * BS + bi_i] != -1)

                # Compute attention scores
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i], bz,
                                              d_i]

                T.gemm(
                    Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i],
                                                   bz, D + d_i]
                T.gemm(
                    Q_tail_shared,
                    KV_tail_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.exp(acc_p[h_i, bi_i] * sm_scale -
                                             Lse[bos + s_i, bz * padded_H + h_i])

                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared,
                    KV_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True)

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (
                        acc_dp[h_i, bi_i] - Delta[bos + s_i, bz * padded_H + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True)
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol)

                T.clear(acc_dkv_tail)
                T.gemm(
                    dP_shared_cast,
                    Q_tail_shared,
                    acc_dkv_tail,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i,
                                                d_i] = acc_dkv_tail[bi_i + s * (BS // split_store),
                                                                    d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i + s *
                                              (BS // split_store)], bz, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4])

                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i + s *
                                              (BS // split_store)], bz, D + d_i * 4],
                            acc_dkv_tail_shared[bi_i, d_i * 4])

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[bos + s_i, bz * padded_H:(bz + 1) * padded_H, :D])
            T.copy(dQ_tail_shared, dQ[bos + s_i, bz * padded_H:(bz + 1) * padded_H, D:])

    return sparse_mla_bwd_kernel


def sparse_mla_bwd(q,
                   kv,
                   o,
                   do,
                   indices,
                   lse,
                   offsets,
                   sm_scale=None,
                   is_casual=True,
                   return_kernel=False,
                   delta=None):
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    S, H, dim_plus_tail_dim = q.shape
    S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert S == S_kv
    # dim should be assigned
    D = 512

    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    assert indices.shape == (S, kv_group, topk)
    assert lse.shape == (S, H)

    token_indices = prepare_token_indices(offsets)

    # Get kernels
    preprocess_kernel = preprocess(H, D)
    bwd_kernel = bwd(H, D, D_tail, topk, kv_group, sm_scale, is_casual)
    postprocess_kernel = postprocess(D, D_tail, kv_group)

    if delta is None:
        delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, do, indices, lse, delta, offsets, token_indices, dkv)
    dkv = postprocess_kernel(dkv)

    return dq, dkv


class TransDSAFunction(torch.autograd.Function):

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
        topk_indices, _ = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
        o, lse = sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, sm_scale=sm_scale, d_v=dim_v)
        ctx.save_for_backward(q, kv, index_q, index_k, weights, topk_indices, o, lse, offsets)
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
        q, kv, index_q, index_k, weights, topk_indices, o, lse, offsets = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(
            q,
            kv.unsqueeze(-2),
            o,
            do,
            topk_indices.unsqueeze(-2),
            lse,
            offsets,
            sm_scale=ctx.sm_scale)
        return dq, dkv.squeeze(-2), None, None, None, None, None, None, None
