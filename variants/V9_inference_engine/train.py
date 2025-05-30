#!/usr/bin/env python3
"""
Convert a fine-tuned ModernBERT checkpoint to ONNX, run O2 graph
optimisation, then quantise weights to 4-bit.

Example
    python quantize_int4.py \
        --model_dir results/20250530_101530/model \
        --out_dir   results/20250530_101530/onnx
"""

import os
from pathlib import Path

from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTOptimizer,
)
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils,
)

# --------------------------------------------------------------------------- #
def export_and_optimize(model_dir: Path, onnx_dir: Path) -> Path:
    """
    • exports the HF checkpoint to ONNX (opset-21, dynamic axes)
    • applies O2 graph-level optimisation
    """
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # 1) Export
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, export=True, opset=21
    )
    ort_model.save_pretrained(onnx_dir)

    # 2) Optimise
    optimizer      = ORTOptimizer.from_pretrained(ort_model)
    optimisation   = AutoOptimizationConfig.O2()          # fuses GELU, LayerNorm, Attention …
    optimised_path = optimizer.optimize(
        save_dir=onnx_dir / "model_optimized",
        optimization_config=optimisation,
    )
    return Path(optimised_path) / "model.onnx"


def quantize_int4(fp32_onnx: Path, out_path: Path, block=128):
    """
    Weight-only 4-bit (symmetric-signed) quantisation of MatMul / Gather.
    Produces  ✓ model_int4.onnx     ✓ model_int4.onnx.data
    """
    cfg = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size      = block,             # power of two, ≥16
        is_symmetric    = True,              # Int4 (signed). False -> UInt4
        accuracy_level  = 4,                 # see MatMulNBits contrib-op docs
        quant_format    = quant_utils.QuantFormat.QOperator,
        op_types_to_quantize = ("MatMul", "Gather"),
        quant_axes      = (("MatMul", 0), ("Gather", 1)),
    )

    model = quant_utils.load_model_with_shape_infer(fp32_onnx)
    quant  = matmul_4bits_quantizer.MatMul4BitsQuantizer(model,
                                                         algo_config=cfg)
    quant.process()

    # external data = TRUE  ➜ large weights stored alongside .onnx
    q_path = out_path / "model_int4.onnx"
    quant.model.save_model_to_file(q_path, True)
    return q_path


# --------------------------------------------------------------------------- #
def main(model_dir: str, out_dir: str):
    model_dir = Path(model_dir)
    out_dir   = Path(out_dir)

    fp32_onnx = export_and_optimize(model_dir, out_dir)
    print(f"✓ exported & optimised: {fp32_onnx}")

    int4_path = quantize_int4(fp32_onnx, out_dir)
    print(f"✓ 4-bit weights saved: {int4_path} (+ .data file)")


if __name__ == "__main__":
    model_dir = "/home/srajput/greenai-pipeline-empirical-study/variants/V0_baseline/results/dummy_20250530_122349"  # Replace with your model directory
    out_dir = "results/onnx"    # Replace with your output directory
    main(model_dir, out_dir)
