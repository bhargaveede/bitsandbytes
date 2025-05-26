from collections.abc import Sequence
import math

import torch

from bitsandbytes.utils import _reverse_4bit_compress_format

from ..._ops import register_kernel
from ..utils import get_habana_frameworks_version


@register_kernel("bitsandbytes::dequantize_4bit", "hpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 or fp4, got {quant_type}")
    torch._check(dtype == torch.bfloat16, lambda: f"4bit dequantization only supports bf16, but got {dtype}")
    torch._check(A.dtype == torch.uint8, lambda: f"quant_storage supports uint8, but got {A.dtype}")

    transpose = False if len(A.shape) == 2 and A.shape[0] == 1 else True

    A = A.reshape(-1)

    habana_version = get_habana_frameworks_version()

    if habana_version.major <= 1 or habana_version.minor < 22:
        _reverse_4bit_compress_format(A)

    # HPU dequantization function for NF4 quantized tensors.
    out_dq = torch.ops.hpu.dequantize_nf4(
        A,
        absmax.to(dtype),
        blocksize,
        out_shape=(math.prod(shape),),
        out_dtype=dtype,
    )

    output = out_dq.reshape(shape)

    if transpose:
        output = output.t()

    return output
