import math
from collections.abc import Sequence
import ctypes as ct

import torch

from bitsandbytes.functional import get_ptr

from ..._ops import register_kernel
from ...cextension import lib


_NF4_QUANT_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
    device="hpu",
)


def _quantize_4bit_impl(
    A: torch.Tensor,
    blocksize: int,
    quant_type: str,
    code: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    torch._check(quant_type in ["nf4", "int8"], lambda: f"4-bit quantization data type {quant_type} is not implemented for HPU.")
    torch._check_is_size(blocksize)

    n = A.numel()
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0

    absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)
    out = torch.zeros(((n + 1) // 2), dtype=torch.uint8, device=A.device)

    rem = n % blocksize
    has_rem = rem > 0

    # Scale tensor to [-1, 1]
    A_reshaped = A.reshape(n)
    A_com = A_reshaped[: n - rem]
    A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
    scaled_A = scaled_A.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)
    
    out_uint8 = torch.empty(scaled_A.shape, dtype=torch.uint8, device=A.device)

    # map [-1, 1] to nf4
    code = code.to(device=scaled_A.device)
    diff = torch.abs(scaled_A.unsqueeze(-1) - code)
    out_uint8 = torch.argmin(diff, dim=-1).to(torch.uint8).to(scaled_A.device)

    if quant_type == "int8":
        out = out_uint8
    else:
        if out_uint8.size(-1) % 2:
            out_uint8 = torch.nn.functional.pad(out_uint8, (0, 1), value=0)
        # To align with HPU dequantize operator
        out[:] = out_uint8[1::2].bitwise_left_shift(4).bitwise_or_(out_uint8[::2])
           
    return out, absmax


@register_kernel("bitsandbytes::quantize_blockwise", "hpu")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    return _quantize_4bit_impl(A, blocksize, "int8", code)


@register_kernel("bitsandbytes::quantize_4bit", "hpu")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    
    torch._check(quant_storage == torch.uint8, lambda: f"quant_storage must be torch.uint8 on HPU, got {quant_storage}")
    return _quantize_4bit_impl(A, blocksize, quant_type, code=_NF4_QUANT_TABLE)

 
@register_kernel("bitsandbytes::dequantize_4bit", "hpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:

    """
    HPU dequantization function for NF4 quantized tensors.
    """
    out_shape = (math.prod(shape),)
    out_dq = torch.ops.hpu.dequantize_nf4(
        A, absmax.to(dtype), blocksize, out_shape=out_shape, out_dtype=dtype
    )
    output = out_dq.reshape(shape).T

    return output
