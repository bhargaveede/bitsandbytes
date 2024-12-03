from typing import Literal, Optional, Tuple, Union

import torch

from bitsandbytes.utils import QuantState

from .base import Backend
from .cpu_xpu_common import (
    dequantize_4bit_impl,
    gemm_4bit_impl,
    igemmlt_impl,
    mm_dequant_impl,
    quantize_4bit_impl,
)

Tensor = torch.Tensor


class HPUBackend(Backend):
    mm_dequant_compute_dtype = torch.bfloat16
    mm_dequant_output_dtype = torch.bfloat16

    def transform(
        self,
        A: torch.Tensor,
        to_order: str,
        from_order="row",
        out: Optional[torch.Tensor] = None,
        transpose=False,
        state: Optional[Tuple[torch.Size, str]] = None,
        ld=None,
    ):
        """
        Transform tensor A to to_order. It is originally designed for CUDA.
        For HPU, it returns the original tensor if transpose=False.
        Otherwise, it returns the transpose of A
        """
        if transpose:
            if out is not None:
                out.copy_(A.T)
            else:
                out = A.T
        else:
            if out is not None:
                out.copy_(A)
            else:
                out = A
        return out, state

    def igemmlt(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        SA: Tuple[torch.Size, str],
        SB: Tuple[torch.Size, str],
        out: Optional[torch.Tensor] = None,
        Sout: Optional[Tuple[torch.Size, str]] = None,
        dtype=torch.int32,
    ) -> Union[torch.Tensor, Tuple[Optional[Tuple[torch.Tensor, Tuple[torch.Size,
                                                                      str]]]]]:

        return igemmlt_impl(A, B, SA, SB, out, Sout, dtype)

    def mm_dequant(
        self,
        A: torch.Tensor,
        quant_state: Tuple[torch.Size, str],
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        new_row_stats: Optional[torch.Tensor] = None,
        new_col_stats: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        return mm_dequant_impl(
            A,
            quant_state,
            row_stats,
            col_stats,
            out,
            new_row_stats,
            new_col_stats,
            bias,
            self.mm_dequant_compute_dtype,
            self.mm_dequant_output_dtype,
        )

    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract columns of A by idx
        """

        return A[:, idx].contiguous()

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        
        if blocksize is None:
            blocksize = 64
        assert quant_storage == torch.uint8
        return quantize_4bit_impl(
            A, absmax, out, blocksize, compress_statistics, quant_type)

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
    
        if blocksize is None:
            blocksize = 64
        return dequantize_4bit_impl(A, quant_state, absmax, out, blocksize, quant_type)

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:

        if state is None:
            raise ValueError(
                "state cannot be None. gemv_4bit() requires the state from quantize_4bit()"
            )

        return gemm_4bit_impl(A, B, out, transposed_A, transposed_B, state)