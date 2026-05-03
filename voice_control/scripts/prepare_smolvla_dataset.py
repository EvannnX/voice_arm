#!/usr/bin/env python3
"""
Validate a local LeRobot dataset and print SmolVLA-ready training arguments.

Usage:
  python voice_control/scripts/prepare_smolvla_dataset.py \
    --dataset-root /home/ima/.cache/huggingface/lerobot/ima/so101_grasp_cup \
    --repo-id ima/so101_grasp_cup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Local dataset root (contains meta/data/videos).")
    parser.add_argument("--repo-id", required=True, help="HF repo id used for training, e.g. ima/so101_grasp_cup")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[ERROR] Missing dataset metadata: {info_path}")
        print("Record dataset first, or pass the correct --dataset-root.")
        return 1

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]

    print("=== Dataset Summary ===")
    print(f"root: {root}")
    print(f"repo_id: {args.repo_id}")
    print(f"episodes: {info.get('total_episodes')}")
    print(f"frames: {info.get('total_frames')}")
    print("video keys:")
    for k in video_keys:
        print(f"  - {k}")

    # Build rename_map for SmolVLA base convention:
    # observation.images.image  (main / agent view)
    # observation.images.image2 (wrist view)
    # Works with both legacy front-only and new agent+wrist naming.
    rename_map: dict[str, str] = {}
    if "observation.images.agent" in video_keys:
        rename_map["observation.images.agent"] = "observation.images.image"
    elif "observation.images.front" in video_keys:
        rename_map["observation.images.front"] = "observation.images.image"

    if "observation.images.wrist" in video_keys:
        rename_map["observation.images.wrist"] = "observation.images.image2"

    print("\n=== SmolVLA Rename Map ===")
    if rename_map:
        rename_map_str = json.dumps(rename_map, ensure_ascii=True)
        print(rename_map_str)
    else:
        rename_map_str = "{}"
        print("{}")
        print("WARNING: no camera mapping generated. Check dataset video keys.")

    # Quick sanity checks
    required_non_visual = ["observation.state", "action"]
    missing_non_visual = [k for k in required_non_visual if k not in features]
    if missing_non_visual:
        print("\n[ERROR] Missing required features for training:")
        for k in missing_non_visual:
            print(f"  - {k}")
        return 2

    if "observation.images.image" in rename_map.values() and "observation.images.image2" not in rename_map.values():
        print("\n[WARN] Only one camera mapped. SmolVLA can still train, but dual-camera is recommended.")

    print("\n=== Train Command (template) ===")
    print("cd /home/ima/Desktop/ITR_LeRobot/lerobot")
    print("source /home/ima/Desktop/ITR_LeRobot/.venv/bin/activate")
    print(
        "lerobot-train "
        "--policy.path=lerobot/smolvla_base "
        f"--dataset.repo_id={args.repo_id} "
        "--batch_size=16 "
        "--steps=20000 "
        "--output_dir=outputs/train/so101_smolvla "
        "--job_name=so101_smolvla "
        "--policy.device=cuda "
        f"--rename_map='{rename_map_str}'"
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

