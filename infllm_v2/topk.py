from __future__ import annotations

from typing import Tuple

import torch

from ._cuda_ext import C


def topk(x: torch.Tensor, top: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last dimension.

    Uses CUDA extension for fp16/bf16 CUDA tensors, otherwise falls back to
    torch.topk. Returns (values, indices).
    """
    if x.dim() < 1:
        raise AssertionError("x must have at least 1 dimension")
    if top <= 0:
        raise AssertionError("top must be > 0")

    dim = x.shape[-1]
    if top > dim:
        raise AssertionError("top cannot be larger than the last dimension")

    # Fallback path.
    if C is None or (not x.is_cuda) or x.dtype not in (torch.float16, torch.bfloat16):
        return torch.topk(x, k=top, dim=-1, largest=True, sorted=True)

    x_contig = x.contiguous()
    leading_shape = x_contig.shape[:-1]
    num_tokens = int(torch.tensor(leading_shape).prod().item()) if len(leading_shape) > 0 else 1
    flat_x = x_contig.reshape(num_tokens, dim)

    values = torch.empty((num_tokens, top), device=x.device, dtype=x.dtype)
    indices = torch.empty((num_tokens, top), device=x.device, dtype=torch.int32)

    dtype_flag = 1 if x.dtype == torch.bfloat16 else 0
    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    C.topk(
        stream,
        num_tokens,
        dim,
        top,
        dtype_flag,
        flat_x.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
    )

    return values.reshape(*leading_shape, top), indices.reshape(*leading_shape, top)
