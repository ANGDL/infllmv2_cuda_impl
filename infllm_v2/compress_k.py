import torch

def round_multiple(x, m):
    return (x + m - 1) // m * m


def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require compression, with stride support.
    """
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
    
    if max_seq_len < chunk_size:
        filtered_indices = torch.tensor([], dtype=torch.long, device=cu_seqlen.device)
        cu_seqlens_compressed = torch.zeros(len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device)
        return filtered_indices, cu_seqlens_compressed
    
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]

    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))

    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    
    chunk_indices = torch.arange(0, chunk_size, device=cu_seqlen.device)[None, :]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices
    filtered_indices = filtered_indices.view(-1)

    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device)
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    
    return filtered_indices, cu_seqlens_compressed


def compress_tensor(tensor, cu_seqlens, kernel_size, kernel_stride, n_heads, head_dim):
    """
    Compress a tensor using mean pooling (CompressK).
    """
    tensor = tensor.transpose(0, 1).contiguous()
    
    filtered_indices, cu_seqlens_compressed = calc_chunks_with_stride(
        cu_seqlens, kernel_size, kernel_stride
    )
    
    if filtered_indices.numel() == 0:
        compressed_tensor = torch.empty(n_heads, 0, head_dim, dtype=tensor.dtype, device=tensor.device)
        return compressed_tensor, cu_seqlens_compressed
    
    filtered_tensor = tensor.index_select(0, filtered_indices.view(-1))
    
    filtered_tensor = filtered_tensor.view(
        filtered_tensor.shape[0] // kernel_size, kernel_size, n_heads, head_dim
    )
    
    compressed_tensor = filtered_tensor.mean(dim=1)
    compressed_tensor = compressed_tensor.transpose(0, 1).contiguous()
    
    return compressed_tensor, cu_seqlens_compressed
