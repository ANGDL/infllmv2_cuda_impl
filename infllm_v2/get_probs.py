from __future__ import annotations

import torch

from ._cuda_ext import C


def get_probs(attn_probs: torch.Tensor, lse: torch.Tensor, scale: float, inplace: bool = False) -> torch.Tensor:
    """Apply exp(attn_probs * scale - lse) row-wise on the last dimension.

    The CUDA kernel updates ``attn_probs`` in place; this wrapper keeps that
    behavior when ``inplace=True`` and returns a new tensor otherwise.
    """
    if attn_probs.dim() < 1:
        raise AssertionError("attn_probs must have at least 1 dimension")
    if lse.dtype != torch.float32:
        lse = lse.to(torch.float32)

    out = attn_probs if inplace else attn_probs.clone()
    dim = out.shape[-1]
    n = int(out.numel() // dim)

    lse_flat = lse.reshape(-1)
    if lse_flat.numel() != n:
        raise AssertionError("lse size must match attn_probs rows (flattened over leading dims)")

    # Fallback path.
    if C is None or (not out.is_cuda) or out.dtype not in (torch.float16, torch.bfloat16):
        out_2d = out.reshape(n, dim)
        out_2d.copy_(torch.exp(out_2d * scale - lse_flat[:, None].to(out_2d.dtype)))
        return out

    out_2d = out.contiguous().reshape(n, dim)
    # Keep lse contiguous on device.
    if lse_flat.device != out.device:
        lse_flat = lse_flat.to(out.device)
    lse_flat = lse_flat.contiguous()

    dtype_flag = 1 if out.dtype == torch.bfloat16 else 0
    stream = torch.cuda.current_stream(device=out.device).cuda_stream
    C.get_probs(
        stream,
        n,
        dim,
        dtype_flag,
        out_2d.data_ptr(),
        lse_flat.data_ptr(),
        float(scale),
    )
    return out_2d.reshape_as(out)
