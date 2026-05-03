#!/usr/bin/env python3
"""
Build SmolVLA-ready train/val dataloaders from a local LeRobot dataset.

This script does light data processing for fine-tuning:
1) validates required keys
2) applies a stable camera rename map (agent->image, wrist->image2)
3) creates episode-level train/val split
4) builds dataloaders with temporal windows (obs history + future action chunk)
5) saves split + config for reproducibility
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_rename_map(feature_keys: list[str]) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    if "observation.images.agent" in feature_keys:
        rename_map["observation.images.agent"] = "observation.images.image"
    elif "observation.images.front" in feature_keys:
        rename_map["observation.images.front"] = "observation.images.image"

    if "observation.images.wrist" in feature_keys:
        rename_map["observation.images.wrist"] = "observation.images.image2"
    return rename_map


def split_episodes(num_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    all_eps = list(range(num_episodes))
    rng = random.Random(seed)
    rng.shuffle(all_eps)
    val_count = max(1, int(round(num_episodes * val_ratio)))
    val_eps = sorted(all_eps[:val_count])
    train_eps = sorted(all_eps[val_count:])
    return train_eps, val_eps


def make_dataset(
    repo_id: str,
    root: Path,
    episodes: list[int],
    action_chunk_size: int,
    n_obs_steps: int,
) -> LeRobotDataset:
    # Use fixed frame deltas for stable training samples.
    # - camera/state history: n_obs_steps ending at current frame
    # - action future chunk: current frame + future steps
    # NOTE: Assumes dataset fps=30 for these defaults.
    dt = 1.0 / 30.0
    obs_deltas = [-(n_obs_steps - 1 - i) * dt for i in range(n_obs_steps)]
    action_deltas = [i * dt for i in range(action_chunk_size)]
    delta_timestamps = {
        "observation.state": obs_deltas,
        "action": action_deltas,
    }
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )


def make_collate_fn(rename_map: dict[str, str]):
    def _collate(batch: list[dict]) -> dict:
        out: dict[str, object] = {}
        keys = batch[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch]
            try:
                out[key] = torch.stack(values)  # tensor-like
            except Exception:
                out[key] = values  # task strings etc.

        # Add renamed camera keys for SmolVLA convention.
        for src, dst in rename_map.items():
            if src in out:
                out[dst] = out[src]
        return out

    return _collate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Local dataset root (contains meta/data/videos).")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. ima/so101_grasp_cup")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--save-config", default="voice_control/data/smolvla_dataloader_config.json")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[ERROR] Missing metadata: {info_path}")
        return 1

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    feature_keys = list(features.keys())

    for key in ("observation.state", "action"):
        if key not in features:
            print(f"[ERROR] Missing required key: {key}")
            return 2

    rename_map = build_rename_map(feature_keys)
    if "observation.images.image" not in rename_map.values():
        print("[ERROR] No main camera key found (agent/front).")
        return 3

    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes < 2:
        print(f"[ERROR] Not enough episodes: {total_episodes}")
        return 4

    train_eps, val_eps = split_episodes(total_episodes, args.val_ratio, args.seed)

    train_ds = make_dataset(args.repo_id, root, train_eps, args.action_chunk_size, args.n_obs_steps)
    val_ds = make_dataset(args.repo_id, root, val_eps, args.action_chunk_size, args.n_obs_steps)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(rename_map),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(rename_map),
    )

    # Dry-run one batch so user can verify shape and key names quickly.
    train_batch = next(iter(train_loader))
    image_keys = sorted([k for k in train_batch.keys() if k.startswith("observation.images.")])

    print("=== SmolVLA DataLoader Ready ===")
    print(f"repo_id: {args.repo_id}")
    print(f"dataset_root: {root}")
    print(f"episodes total/train/val: {total_episodes}/{len(train_eps)}/{len(val_eps)}")
    print(f"frames train/val: {train_ds.num_frames}/{val_ds.num_frames}")
    print(f"rename_map: {json.dumps(rename_map, ensure_ascii=True)}")
    print(f"image keys in batch: {image_keys}")
    print(f"state shape: {tuple(train_batch['observation.state'].shape)}")
    print(f"action shape: {tuple(train_batch['action'].shape)}")
    if "task" in train_batch:
        print(f"task sample: {train_batch['task'][0]}")

    save_path = Path(args.save_config).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "dataset_root": str(root),
        "repo_id": args.repo_id,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "action_chunk_size": args.action_chunk_size,
        "n_obs_steps": args.n_obs_steps,
        "rename_map": rename_map,
        "train_episodes": train_eps,
        "val_episodes": val_eps,
    }
    save_path.write_text(json.dumps(save_payload, ensure_ascii=True, indent=2))
    print(f"saved config: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
