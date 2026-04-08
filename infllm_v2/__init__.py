__version__ = "0.1.0"

from .blockmask_to_uint64 import blockmask_to_uint64
from .topk_to_uint64 import topk_to_uint64
from .uint64_to_bool import uint64_to_bool
from .max_pooling_1d import max_pooling_1d, max_pooling_1d_varlen
from .topk import topk
from .get_probs import get_probs

try:
    from .infllmv2_sparse_attention import (
        infllmv2_attn_varlen_func,
        infllmv2_attn_stage1,
        infllmv2_attn_with_kvcache,
    )
except Exception:
    infllmv2_attn_varlen_func = None
    infllmv2_attn_stage1 = None
    infllmv2_attn_with_kvcache = None
