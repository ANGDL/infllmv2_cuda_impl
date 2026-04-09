import torch

from infllm_v2.infllmv2_sparse_attention import (
    infllmv2_attn_stage1,
    infllmv2_attn_varlen_func,
)


def _repeat_kv_heads(x: torch.Tensor, nheads_q: int) -> torch.Tensor:
    nheads_k = x.shape[1]
    if nheads_q == nheads_k:
        return x
    return x.repeat_interleave(nheads_q // nheads_k, dim=1)


def _ref_varlen_attn(q, k, v, cu_q, cu_k, causal, scale):
    total_q, nheads_q, _ = q.shape
    k_full = _repeat_kv_heads(k, nheads_q)
    v_full = _repeat_kv_heads(v, nheads_q)
    out = torch.zeros_like(q)

    for b in range(cu_q.numel() - 1):
        q_start = int(cu_q[b].item())
        q_end = int(cu_q[b + 1].item())
        k_start = int(cu_k[b].item())
        k_end = int(cu_k[b + 1].item())
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = q[q_start:q_end].transpose(0, 1)
        k_b = k_full[k_start:k_end].transpose(0, 1)
        v_b = v_full[k_start:k_end].transpose(0, 1)
        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale

        if causal:
            q_idx = torch.arange(q_len)[:, None]
            k_idx = torch.arange(k_len)[None, :]
            mask = k_idx > (q_idx + k_len - q_len)
            scores = scores.masked_fill(mask.to(scores.device)[None, :, :], -float("inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        out[q_start:q_end] = torch.matmul(probs, v_b).transpose(0, 1)

    return out


def _ref_stage1(q, k, cu_q, cu_k, max_seqlen_k, causal, scale):
    total_q, nheads_q, _ = q.shape
    _, nheads_k, _ = k.shape
    group_size = nheads_q // nheads_k
    out = torch.zeros((nheads_k, total_q, max_seqlen_k), dtype=q.dtype)

    for b in range(cu_q.numel() - 1):
        q_start = int(cu_q[b].item())
        q_end = int(cu_q[b + 1].item())
        k_start = int(cu_k[b].item())
        k_end = int(cu_k[b + 1].item())
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = q[q_start:q_end].transpose(0, 1)
        k_b = k[k_start:k_end].transpose(0, 1)
        k_rep = k_b.repeat_interleave(group_size, dim=0)
        scores = torch.matmul(q_b, k_rep.transpose(-2, -1)) * scale

        if causal:
            q_idx = torch.arange(q_len)
            right = ((q_idx - 15) // 16) + k_len - (q_len - 16 + 1) // 16
            right = right.clamp(0, k_len)
            k_idx = torch.arange(k_len)[None, :]
            mask = k_idx >= right[:, None]
            scores = scores.masked_fill(mask.to(scores.device)[None, :, :], -float("inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = probs.reshape(nheads_k, group_size, q_len, k_len).sum(dim=1)
        out[:, q_start:q_end, :k_len] = probs

    return out


def test_infllmv2_attn_varlen_func_torch_fallback_matches_reference_cpu():
    torch.manual_seed(0)
    q = torch.randn(7, 4, 8, dtype=torch.float32)
    k = torch.randn(9, 2, 8, dtype=torch.float32)
    v = torch.randn(9, 2, 8, dtype=torch.float32)
    cu_q = torch.tensor([0, 3, 7], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 9], dtype=torch.int32)
    scale = q.shape[-1] ** (-0.5)

    out = infllmv2_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=4,
        max_seqlen_k=5,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        return_attn_probs=False,
    )
    ref = _ref_varlen_attn(q, k, v, cu_q, cu_k, causal=True, scale=scale)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_infllmv2_attn_stage1_torch_fallback_matches_reference_cpu():
    torch.manual_seed(1)
    q = torch.randn(12, 4, 8, dtype=torch.float32)
    k = torch.randn(10, 2, 8, dtype=torch.float32)
    v = torch.randn(10, 2, 8, dtype=torch.float32)
    cu_q = torch.tensor([0, 5, 12], dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 10], dtype=torch.int32)
    scale = q.shape[-1] ** (-0.5)

    out = infllmv2_attn_stage1(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_seqlens_v=cu_k,
        max_seqlen_q=7,
        max_seqlen_k=10,
        causal=True,
    )
    ref = _ref_stage1(q, k, cu_q, cu_k, max_seqlen_k=10, causal=True, scale=scale)
    assert torch.allclose(out[:, :, :10], ref[:, :, :10], atol=1e-5, rtol=1e-5)
