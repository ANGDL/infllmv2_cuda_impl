import torch

from infllm_v2 import (
    blockmask_to_uint64,
    uint64_to_bool,
    topk_to_uint64,
    max_pooling_1d,
    max_pooling_1d_varlen,
    topk,
    get_probs,
)


def test_blockmask_uint64_roundtrip_cpu():
    mask = torch.randint(0, 2, (2, 3, 17), dtype=torch.bool)
    packed, last_dim = blockmask_to_uint64(mask)
    recovered = uint64_to_bool(packed, last_dim)
    assert torch.equal(mask, recovered)


def test_topk_to_uint64_cpu_matches_manual_mask():
    idx = torch.tensor([[[0, 1, -1], [1, 0, 1]]], dtype=torch.int32)
    packed, k_blocks = topk_to_uint64(idx, max_seqlen_k=128, block_size=64)

    expected_mask = torch.zeros((1, 2, k_blocks), dtype=torch.bool)
    expected_mask[0, 0, 0] = True
    expected_mask[0, 0, 1] = True
    expected_mask[0, 1, 0] = True
    expected_mask[0, 1, 1] = True
    expected_packed, _ = blockmask_to_uint64(expected_mask)
    assert torch.equal(packed, expected_packed)


def test_pooling_shapes_cpu():
    x = torch.randn(2, 8, 16, dtype=torch.float16)
    y = max_pooling_1d(x, cache_len=0, local_blocks=1, init_blocks=1, block_size=4, stride=2)
    assert y.shape == (2, 8, 2)

    xv = torch.randn(2, 6, 10, dtype=torch.float16)
    cu_q = torch.tensor([0, 2, 6], dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 10], dtype=torch.int32)
    cache_lens = torch.tensor([0, 1], dtype=torch.int32)
    yv = max_pooling_1d_varlen(
        xv,
        cu_q,
        cu_k,
        cache_lens,
        max_seqlen_q=4,
        max_context_len=10,
        local_blocks=1,
        init_blocks=1,
        block_size=4,
        stride=2,
    )
    assert yv.shape == (2, 6, 2)


def test_topk_fallback_cpu_matches_torch_topk():
    x = torch.randn(3, 7, dtype=torch.float32)
    v, i = topk(x, top=3)
    v_ref, i_ref = torch.topk(x, k=3, dim=-1, largest=True, sorted=True)
    assert torch.allclose(v, v_ref)
    assert torch.equal(i, i_ref)


def test_get_probs_fallback_cpu_formula():
    probs = torch.randn(2, 5, dtype=torch.float32)
    lse = torch.randn(2, dtype=torch.float32)
    scale = 0.7

    out = get_probs(probs, lse, scale)
    ref = torch.exp(probs * scale - lse[:, None])
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
