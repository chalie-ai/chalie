#!/usr/bin/env python3
"""
Pre-optimize ONNX models for faster cold boot.

Loads each model with ORT_ENABLE_EXTENDED graph optimization and saves the
optimized graph to disk. At runtime, services detect these files and
skip graph optimization entirely — turning a multi-minute session
creation into a sub-second file read.

Uses ORT_ENABLE_EXTENDED (not ORT_ENABLE_ALL) so the output is portable
across CPU architectures (ARM/x86). The runtime fallback still uses
ORT_ENABLE_ALL for hardware-specific optimizations when no pre-optimized
file exists.

The optimized files are tied to the pinned onnxruntime version (1.20.1).
Re-run this script after upgrading onnxruntime.

Usage:
    cd backend && python scripts/optimize_models.py

    # Then upload the .optimized.onnx files to GitHub releases or
    # commit them to the models directory for Docker builds.
"""

import sys
import time
from pathlib import Path

# Ensure backend/ is on path
_backend = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend))

import onnxruntime as ort  # noqa: E402

MODELS_DIR = _backend / "data" / "models"


def optimize_model(onnx_path: Path, optimized_path: Path,
                   intra_threads: int = 2) -> bool:
    """Optimize a single ONNX model and save the result."""
    if not onnx_path.exists():
        print(f"  SKIP  {onnx_path} (not found)")
        return False

    if optimized_path.exists():
        raw_mtime = onnx_path.stat().st_mtime
        opt_mtime = optimized_path.stat().st_mtime
        if opt_mtime > raw_mtime:
            size_mb = optimized_path.stat().st_size / (1024 * 1024)
            print(f"  OK    {optimized_path.name} ({size_mb:.0f}MB, already up to date)")
            return True

    raw_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  OPT   {onnx_path.name} ({raw_mb:.0f}MB) ...", end=" ", flush=True)

    # Write to temp file, then atomic rename to avoid corrupt partial files
    tmp_path = optimized_path.with_suffix(".onnx.tmp")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    # EXTENDED is portable across CPU architectures (no NchwcTransformer).
    # Runtime fallback uses ORT_ENABLE_ALL for hardware-specific gains.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opts.optimized_model_filepath = str(tmp_path)

    t0 = time.time()
    ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    elapsed = time.time() - t0

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        print("FAIL (ORT produced no output)")
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.rename(optimized_path)
    opt_mb = optimized_path.stat().st_size / (1024 * 1024)
    print(f"{opt_mb:.0f}MB in {elapsed:.1f}s")
    return True


def main():
    print(f"onnxruntime {ort.__version__}")
    print(f"Models dir: {MODELS_DIR}\n")

    results = []

    # 1. Embedding model (gte-modernbert-base)
    print("[1/2] Embedding model (gte-modernbert-base)")
    emb_raw = MODELS_DIR / "gte-modernbert-base" / "onnx" / "model.onnx"
    emb_opt = emb_raw.with_suffix(".optimized.onnx")
    results.append(("embedding", optimize_model(emb_raw, emb_opt, intra_threads=2)))

    # 2. ONNX classifier base (Qwen2.5-0.5B)
    print("[2/2] Classifier base model (Qwen2.5-0.5B)")
    cls_raw = MODELS_DIR / "qwen2.5-0.5b_base" / "model.onnx"
    cls_opt = cls_raw.with_suffix(".optimized.onnx")
    results.append(("classifier-base", optimize_model(cls_raw, cls_opt, intra_threads=1)))

    print()
    for name, ok in results:
        status = "OK" if ok else "SKIPPED"
        print(f"  {name}: {status}")

    if all(ok for _, ok in results):
        print("\nAll models optimized. Upload .optimized.onnx files to GitHub releases.")
    else:
        print("\nSome models skipped — download them first (run Chalie once).")


if __name__ == "__main__":
    main()
