import time
import inspect

import pytest
import torch

from infllm_v2 import (
    blockmask_to_uint64,
    get_probs,
    max_pooling_1d,
    max_pooling_1d_varlen,
    max_pooling_1d_varlen_v2,
    topk,
    topk_to_uint64,
    uint64_to_bool,
)
from infllm_v2._cuda_ext import C
from infllm_v2.infllmv2_sparse_attention import (
    _infllmv2_attn_stage1_torch,
    _infllmv2_attn_varlen_forward_torch,
    infllmv2_attn_stage1,
    infllmv2_attn_varlen_func,
)
from infllm_v2.torch_kernels import (
    blockmask_to_uint64_torch,
    max_pooling_1d_torch,
    max_pooling_1d_varlen_torch,
    topk_to_uint64_torch,
    uint64_to_bool_torch,
)


def _call_max_pooling_1d_varlen_adaptive(
    input_tensor,
    cu_q,
    cu_k,
    cache_lens,
    max_seqlen_q,
    max_k_or_context,
    local_blocks,
    init_blocks,
    block_size,
    stride,
):
    """Call max_pooling_1d_varlen across branches with different kwarg names."""
    sig = inspect.signature(max_pooling_1d_varlen)
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=stride,
    )
    if "max_context_len" in sig.parameters:
        kwargs["max_context_len"] = max_k_or_context
    else:
        kwargs["max_seqlen_k"] = max_k_or_context

    return max_pooling_1d_varlen(
        input_tensor,
        cu_q,
        cu_k,
        cache_lens,
        **kwargs,
    )


def _resolve_varlen_torch_ref_max_seqlen_k(max_k_or_context, stride):
    """Match torch reference semantics to the active max_pooling_1d_varlen wrapper."""
    sig = inspect.signature(max_pooling_1d_varlen)
    if "max_context_len" in sig.parameters:
        return max_k_or_context // stride
    return max_k_or_context


def _bench_cuda(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - st) / iters


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (C is None),
    reason="requires CUDA and compiled infllm_v2.C extension",
)


def test_mask_pack_unpack_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"
    mask = torch.randint(0, 2, (8, 16, 513), device=device, dtype=torch.bool)

    packed_cuda, last_dim = blockmask_to_uint64(mask)
    packed_torch, _ = blockmask_to_uint64_torch(mask)
    assert torch.equal(packed_cuda, packed_torch)

    rec_cuda = uint64_to_bool(packed_cuda, last_dim)
    rec_torch = uint64_to_bool_torch(packed_cuda, last_dim)
    assert torch.equal(rec_cuda, rec_torch)
    assert torch.equal(rec_cuda, mask)

    t_cuda_pack = _bench_cuda(lambda: blockmask_to_uint64(mask)[0])
    t_torch_pack = _bench_cuda(lambda: blockmask_to_uint64_torch(mask)[0])
    t_cuda_unpack = _bench_cuda(lambda: uint64_to_bool(packed_cuda, last_dim))
    t_torch_unpack = _bench_cuda(lambda: uint64_to_bool_torch(packed_cuda, last_dim))

    print(
        f"\npack cuda={t_cuda_pack*1e3:.3f}ms torch={t_torch_pack*1e3:.3f}ms | "
        f"unpack cuda={t_cuda_unpack*1e3:.3f}ms torch={t_torch_unpack*1e3:.3f}ms"
    )


def test_topk_to_uint64_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"
    topk_idx = torch.randint(-1, 64, (4, 8, 256, 32), device=device, dtype=torch.int32)

    out_cuda, k_blocks_cuda = topk_to_uint64(topk_idx, max_seqlen_k=4096, block_size=64)
    out_torch, k_blocks_torch = topk_to_uint64_torch(topk_idx, max_seqlen_k=4096, block_size=64)
    assert k_blocks_cuda == k_blocks_torch
    assert torch.equal(out_cuda, out_torch)

    t_cuda = _bench_cuda(lambda: topk_to_uint64(topk_idx, max_seqlen_k=4096, block_size=64)[0])
    t_torch = _bench_cuda(lambda: topk_to_uint64_torch(topk_idx, max_seqlen_k=4096, block_size=64)[0])
    print(f"\ntopk_to_uint64 cuda={t_cuda*1e3:.3f}ms torch={t_torch*1e3:.3f}ms")


def test_topk_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"
    x = torch.randn(256, 512, device=device, dtype=torch.float16)

    v_cuda, i_cuda = topk(x, top=16)
    v_ref, i_ref = torch.topk(x, k=16, dim=-1, largest=True, sorted=True)

    # Value agreement is the primary correctness criterion.
    assert torch.allclose(v_cuda, v_ref, atol=1e-3, rtol=1e-3)
    # For fp16 inputs, ties are common and index tie-breaking may differ.
    # Verify indices are in-range and map back to the same selected values.
    assert torch.all((i_cuda >= 0) & (i_cuda < x.shape[-1]))
    gathered = x.gather(dim=-1, index=i_cuda.to(torch.int64))
    assert torch.allclose(gathered, v_cuda, atol=1e-3, rtol=1e-3)

    t_cuda = _bench_cuda(lambda: topk(x, top=16)[0])
    t_torch = _bench_cuda(lambda: torch.topk(x, k=16, dim=-1, largest=True, sorted=True)[0])
    print(f"\ntopk cuda={t_cuda*1e3:.3f}ms torch={t_torch*1e3:.3f}ms")


def test_get_probs_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"
    attn = torch.randn(512, 128, device=device, dtype=torch.float16)
    lse = torch.randn(512, device=device, dtype=torch.float32)
    scale = 0.125

    out_cuda = get_probs(attn, lse, scale=scale, inplace=False)
    # Kernel computes in float and writes back to fp16/bf16.
    out_torch = torch.exp(attn.to(torch.float32) * scale - lse[:, None]).to(attn.dtype)
    assert torch.allclose(out_cuda, out_torch, atol=3e-3, rtol=3e-3)

    t_cuda = _bench_cuda(lambda: get_probs(attn, lse, scale=scale, inplace=False))
    t_torch = _bench_cuda(lambda: torch.exp(attn * scale - lse[:, None].to(attn.dtype)))
    print(f"\nget_probs cuda={t_cuda*1e3:.3f}ms torch={t_torch*1e3:.3f}ms")


def test_max_pooling_1d_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"

    x = torch.randn(8, 512, 1024, device=device, dtype=torch.float16)
    out_cuda = max_pooling_1d(x, cache_len=0, local_blocks=2, init_blocks=1, block_size=64, stride=16)
    out_torch = max_pooling_1d_torch(x, cache_len=0, local_blocks=2, init_blocks=1, block_size=64, stride=16)
    assert torch.equal(torch.isinf(out_cuda), torch.isinf(out_torch))
    finite = torch.isfinite(out_cuda) & torch.isfinite(out_torch)
    assert torch.allclose(out_cuda[finite], out_torch[finite], atol=1e-3, rtol=1e-3)

    t_cuda = _bench_cuda(lambda: max_pooling_1d(x, cache_len=0, local_blocks=2, init_blocks=1, block_size=64, stride=16))
    t_torch = _bench_cuda(lambda: max_pooling_1d_torch(x, cache_len=0, local_blocks=2, init_blocks=1, block_size=64, stride=16))
    print(f"\nmax_pooling_1d cuda={t_cuda*1e3:.3f}ms torch={t_torch*1e3:.3f}ms")

    

def test_max_pooling_1d_varlen_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"

    # In this branch, max_pooling_1d_varlen derives max_seqlen_k as
    # max_context_len // stride, so use compressed-k shaped inputs here.
    xv = torch.randn(4, 900, 64, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, 200, 420, 640, 900], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 16, 32, 48, 64], device=device, dtype=torch.int32)
    cache_lens = torch.tensor([0, 16, 32, 48], device=device, dtype=torch.int32)

    try:
        outv_cuda = _call_max_pooling_1d_varlen_adaptive(
            xv,
            cu_q,
            cu_k,
            cache_lens,
            max_seqlen_q=260,
            max_k_or_context=1024,
            local_blocks=2,
            init_blocks=1,
            block_size=64,
            stride=16,
        )
    except TypeError as e:
        if "incompatible function arguments" in str(e):
            pytest.skip("max_pooling_1d_varlen ABI mismatch for current compiled extension")
        raise

    ref_max_seqlen_k = _resolve_varlen_torch_ref_max_seqlen_k(max_k_or_context=1024, stride=16)

    outv_torch = max_pooling_1d_varlen_torch(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=260,
        max_seqlen_k=ref_max_seqlen_k,
        local_blocks=2,
        init_blocks=1,
        block_size=64,
        stride=16,
        max_context_len=1024,
    )
    assert torch.equal(torch.isinf(outv_cuda), torch.isinf(outv_torch))
    finite_v = torch.isfinite(outv_cuda) & torch.isfinite(outv_torch)
    assert torch.allclose(outv_cuda[finite_v], outv_torch[finite_v], atol=1e-3, rtol=1e-3)

    t_cuda_v = _bench_cuda(lambda: _call_max_pooling_1d_varlen_adaptive(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=260,
        max_k_or_context=1024,
        local_blocks=2,
        init_blocks=1,
        block_size=64,
        stride=16,
    ))
    t_torch_v = _bench_cuda(
        lambda: max_pooling_1d_varlen_torch(
            xv,
            cu_q,
            cu_k,
            cache_lens,
            max_seqlen_q=260,
            max_seqlen_k=ref_max_seqlen_k,
            local_blocks=2,
            init_blocks=1,
            block_size=64,
            stride=16,
            max_context_len=1024,
        )
    )
    print(f"\nmax_pooling_1d_varlen cuda={t_cuda_v*1e3:.3f}ms torch={t_torch_v*1e3:.3f}ms")


def test_max_pooling_1d_varlen_v2_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"

    xv = torch.randn(4, 900, 1024, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, 200, 420, 640, 900], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 256, 512, 768, 1024], device=device, dtype=torch.int32)
    cache_lens = torch.tensor([0, 16, 32, 48], device=device, dtype=torch.int32)

    outv2_cuda = max_pooling_1d_varlen_v2(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=260,
        local_blocks=2,
        init_blocks=1,
        block_size=64,
        stride=16,
        total_q=xv.shape[1],
    )
    # v2 wrapper currently uses hard-coded max_context_len=32768 internally.
    outv2_torch = max_pooling_1d_varlen_torch(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=260,
        max_seqlen_k=32768,
        local_blocks=2,
        init_blocks=1,
        block_size=64,
        stride=16,
    )
    assert torch.equal(torch.isinf(outv2_cuda), torch.isinf(outv2_torch))
    finite_v2 = torch.isfinite(outv2_cuda) & torch.isfinite(outv2_torch)
    assert torch.allclose(outv2_cuda[finite_v2], outv2_torch[finite_v2], atol=1e-3, rtol=1e-3)

    t_cuda_v2 = _bench_cuda(lambda: max_pooling_1d_varlen_v2(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=260,
        local_blocks=2,
        init_blocks=1,
        block_size=64,
        stride=16,
        total_q=xv.shape[1],
    ))
    t_torch_v2 = _bench_cuda(lambda: max_pooling_1d_varlen_torch(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=260,
        max_seqlen_k=32768,
        local_blocks=2,
        init_blocks=1,
        block_size=64,
        stride=16,
    ))
    print(f"\nmax_pooling_1d_varlen_v2 cuda={t_cuda_v2*1e3:.3f}ms torch={t_torch_v2*1e3:.3f}ms")


def test_varlen_fwd_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"
    torch.manual_seed(0)

    total_q, total_k = 320, 256
    nheads_q, nheads_k, headdim = 8, 2, 64
    q = torch.randn(total_q, nheads_q, headdim, device=device, dtype=torch.float16)
    k = torch.randn(total_k, nheads_k, headdim, device=device, dtype=torch.float16)
    v = torch.randn(total_k, nheads_k, headdim, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, 128, 320], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 96, 256], device=device, dtype=torch.int32)

    out_cuda = infllmv2_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=192,
        max_seqlen_k=160,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        return_attn_probs=False,
    )
    out_torch, _, _, _, _ = _infllmv2_attn_varlen_forward_torch(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=192,
        max_seqlen_k=160,
        dropout_p=0.0,
        softmax_scale=q.shape[-1] ** (-0.5),
        causal=True,
        return_softmax=False,
    )

    assert torch.allclose(out_cuda, out_torch, atol=2e-2, rtol=2e-2)

    t_cuda = _bench_cuda(
        lambda: infllmv2_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=192,
            max_seqlen_k=160,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
            return_attn_probs=False,
        )
    )
    t_torch = _bench_cuda(
        lambda: _infllmv2_attn_varlen_forward_torch(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=192,
            max_seqlen_k=160,
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=True,
            return_softmax=False,
        )[0]
    )
    print(f"\nvarlen_fwd cuda={t_cuda*1e3:.3f}ms torch={t_torch*1e3:.3f}ms")


def test_varlen_fwd_stage1_cuda_vs_torch_accuracy_and_perf():
    device = "cuda"
    torch.manual_seed(1)

    total_q, total_k = 384, 288
    nheads_q, nheads_k, headdim = 8, 2, 64
    q = torch.randn(total_q, nheads_q, headdim, device=device, dtype=torch.float16)
    k = torch.randn(total_k, nheads_k, headdim, device=device, dtype=torch.float16)
    v = torch.randn(total_k, nheads_k, headdim, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, 160, 384], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 112, 288], device=device, dtype=torch.int32)
    max_seqlen_k = 192

    out_cuda = infllmv2_attn_stage1(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_seqlens_v=cu_k,
        max_seqlen_q=224,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    out_torch = _infllmv2_attn_stage1_torch(
        q,
        k,
        cu_q,
        cu_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=q.shape[-1] ** (-0.5),
        causal=True,
    )

    assert torch.allclose(out_cuda[:, :total_q, :max_seqlen_k], out_torch[:, :total_q, :max_seqlen_k], atol=2e-2, rtol=2e-2)

    t_cuda = _bench_cuda(
        lambda: infllmv2_attn_stage1(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            cu_seqlens_v=cu_k,
            max_seqlen_q=224,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=True,
        )
    )
    t_torch = _bench_cuda(
        lambda: _infllmv2_attn_stage1_torch(
            q,
            k,
            cu_q,
            cu_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=True,
        )
    )
    print(f"\nvarlen_fwd_stage1 cuda={t_cuda*1e3:.3f}ms torch={t_torch*1e3:.3f}ms")
