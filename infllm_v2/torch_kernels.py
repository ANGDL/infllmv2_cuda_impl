from __future__ import annotations

from typing import Tuple

import torch


def blockmask_to_uint64_torch(blockmask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if blockmask.dtype != torch.bool:
        blockmask = blockmask.to(torch.bool)

    original_shape = blockmask.shape
    last_dim_size = original_shape[-1]
    n_uint64_per_row = (last_dim_size + 63) // 64

    flat = blockmask.reshape(-1, last_dim_size)
    if n_uint64_per_row * 64 != last_dim_size:
        pad = n_uint64_per_row * 64 - last_dim_size
        flat = torch.nn.functional.pad(flat, (0, pad), value=False)

    flat = flat.view(-1, n_uint64_per_row, 64)
    bit_weights = (torch.ones(64, device=flat.device, dtype=torch.int64) << torch.arange(64, device=flat.device, dtype=torch.int64))
    packed = (flat.to(torch.int64) * bit_weights).sum(dim=-1)

    output_shape = original_shape[:-1] + (n_uint64_per_row,)
    return packed.reshape(output_shape), last_dim_size


def uint64_to_bool_torch(uint64_array: torch.Tensor, last_dim_size: int) -> torch.Tensor:
    x = uint64_array.to(torch.int64)
    n_uint64_per_row = x.shape[-1]

    flat = x.reshape(-1, n_uint64_per_row)
    bits = (torch.arange(64, device=flat.device, dtype=torch.int64))[None, None, :]
    expanded = ((flat[:, :, None] >> bits) & 1).to(torch.bool)
    expanded = expanded.reshape(flat.shape[0], n_uint64_per_row * 64)
    expanded = expanded[:, :last_dim_size]

    output_shape = x.shape[:-1] + (last_dim_size,)
    return expanded.reshape(output_shape)


def topk_to_uint64_torch(topk_idx: torch.Tensor, max_seqlen_k: int, block_size: int) -> Tuple[torch.Tensor, int]:
    if topk_idx.dtype != torch.int32:
        raise AssertionError("topk_idx must be int32")

    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    n_uint64_per_row = (k_blocks + 63) // 64

    original_shape = topk_idx.shape
    if len(original_shape) == 4:
        batch_size, num_heads, total_seqlen, k = original_shape
        flat = topk_idx.reshape(batch_size * num_heads * total_seqlen, k)
        out_shape = (batch_size, num_heads, total_seqlen, n_uint64_per_row)
    elif len(original_shape) == 3:
        num_heads, total_seqlen, k = original_shape
        flat = topk_idx.reshape(num_heads * total_seqlen, k)
        out_shape = (num_heads, total_seqlen, n_uint64_per_row)
    else:
        raise AssertionError("topk_idx must be 3D or 4D")

    bool_mask = torch.zeros(flat.shape[0], k_blocks, dtype=torch.bool, device=flat.device)
    for col in range(flat.shape[1]):
        curr = flat[:, col]
        valid = (curr >= 0) & (curr < k_blocks)
        if valid.any():
            rows = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            bool_mask[rows, curr[rows].to(torch.long)] = True

    packed, _ = blockmask_to_uint64_torch(bool_mask)
    return packed.reshape(out_shape), k_blocks


def _pool_windows(input_tensor: torch.Tensor, out_len: int, k_len: int, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    if k_len == 0:
        return torch.full((*input_tensor.shape[:-1], out_len), -float("inf"), dtype=input_tensor.dtype, device=input_tensor.device)

    base = torch.arange(out_len, device=input_tensor.device, dtype=torch.long)[:, None] * stride - padding
    offs = torch.arange(kernel_size, device=input_tensor.device, dtype=torch.long)[None, :]
    idx = base + offs
    valid = (idx >= 0) & (idx < k_len)
    safe_idx = idx.clamp(0, k_len - 1)

    gathered = input_tensor[:, :, safe_idx]
    neg_inf = torch.tensor(-float("inf"), device=input_tensor.device, dtype=input_tensor.dtype)
    gathered = torch.where(valid[None, None, :, :], gathered, neg_inf)
    return gathered.max(dim=-1).values


def max_pooling_1d_torch(
    input: torch.Tensor,
    cache_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
) -> torch.Tensor:
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise AssertionError("input must be float16 or bfloat16")

    input = input.contiguous()
    num_heads, q_len, k_len = input.shape

    stride_inner = block_size // stride
    kernel_size = stride_inner + 1
    padding = 1
    out_len = (q_len + cache_len + block_size - 1) // block_size

    pooled = _pool_windows(input, out_len, k_len, kernel_size, stride_inner, padding)

    off_bq = (torch.arange(q_len, device=input.device, dtype=torch.long) + cache_len) // block_size
    off_bk = torch.arange(out_len, device=input.device, dtype=torch.long)
    mask = (off_bk[None, :] < init_blocks) | ((off_bq[:, None] >= off_bk[None, :]) & (off_bq[:, None] <= (off_bk[None, :] + local_blocks)))

    pos_inf = torch.tensor(float("inf"), device=input.device, dtype=input.dtype)
    pooled = torch.where(mask[None, :, :], pos_inf, pooled)
    return pooled


def max_pooling_1d_varlen_torch(
    input: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cache_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
) -> torch.Tensor:
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise AssertionError("input must be float16 or bfloat16")

    input = input.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    cache_lens = cache_lens.contiguous()

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = input.shape[0]
    total_q = input.shape[1]

    if cu_seqlens_q[-1].item() != total_q:
        raise AssertionError("total_q mismatch with cu_seqlens_q")
    if input.shape[2] != max_seqlen_k:
        raise AssertionError("max_k mismatch")
    if cache_lens.shape[0] != batch_size:
        raise AssertionError("cache_lens batch size mismatch")

    stride_inner = block_size // stride
    kernel_size = stride_inner + 1
    padding = 1
    max_cache_len = int(cache_lens.max().item()) if cache_lens.numel() > 0 else 0
    out_len = (max_seqlen_q + max_cache_len + block_size - 1) // block_size

    output = torch.zeros(num_heads, total_q, out_len, device=input.device, dtype=input.dtype)
    pos_inf = torch.tensor(float("inf"), device=input.device, dtype=input.dtype)

    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b + 1].item())
        q_len = q_end - q_start
        k_len = k_end - k_start
        cache_len = int(cache_lens[b].item())

        if q_len == 0:
            continue

        inp_b = input[:, q_start:q_end, :k_len]
        pooled = _pool_windows(inp_b, out_len, k_len, kernel_size, stride_inner, padding)

        off_bq = (torch.arange(q_len, device=input.device, dtype=torch.long) + cache_len) // block_size
        off_bk = torch.arange(out_len, device=input.device, dtype=torch.long)
        mask = (off_bk[None, :] < init_blocks) | ((off_bq[:, None] >= off_bk[None, :]) & (off_bq[:, None] <= (off_bk[None, :] + local_blocks)))
        pooled = torch.where(mask[None, :, :], pos_inf, pooled)

        output[:, q_start:q_end, :] = pooled

    return output
