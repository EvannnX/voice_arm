#!/usr/bin/env python3
"""
Prepare a complete local policy directory for SmolVLA evaluation from a single model.safetensors.

It downloads non-weight metadata/processor files from a base model repo (default: lerobot/smolvla_base)
and injects your fine-tuned `model.safetensors`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ALLOW_PATTERNS = [
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor*.safetensors",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a complete local SmolVLA policy dir for lerobot-record.")
    parser.add_argument("--model-file", required=True, help="Path to fine-tuned model.safetensors")
    parser.add_argument(
        "--out-dir",
        default="/home/ima/Desktop/ITR_LeRobot/pretrained_model_ready",
        help="Output policy directory (will be recreated).",
    )
    parser.add_argument(
        "--base-model",
        default="lerobot/smolvla_base",
        help="HF base model repo used to fetch config + processor files.",
    )
    args = parser.parse_args()

    model_file = Path(args.model_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not model_file.exists():
        print(f"[ERROR] model file not found: {model_file}")
        return 1

    print(f"[1/4] Downloading base metadata from {args.base_model} ...")
    snap_dir = Path(
        snapshot_download(
            repo_id=args.base_model,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=["*.bin", "*.pt", "*.onnx", "model.safetensors"],
        )
    )

    print(f"[2/4] Rebuilding output dir: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[3/4] Copying config and processor files ...")
    copied = 0
    for rel in ALLOW_PATTERNS:
        # Expand simple wildcard patterns manually.
        if "*" in rel:
            for src in snap_dir.glob(rel):
                shutil.copy2(src, out_dir / src.name)
                copied += 1
        else:
            src = snap_dir / rel
            if src.exists():
                shutil.copy2(src, out_dir / src.name)
                copied += 1

    print(f"[4/4] Injecting fine-tuned weights: {model_file}")
    shutil.copy2(model_file, out_dir / "model.safetensors")

    required = ["config.json", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors"]
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        print(f"[ERROR] Missing required files after build: {missing}")
        return 2

    print("\n[OK] Policy directory is ready:")
    print(out_dir)
    print(f"[OK] Copied metadata files: {copied}, plus model.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
