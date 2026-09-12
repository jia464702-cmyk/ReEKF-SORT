#!/usr/bin/env python3
"""Patch benchmark YAMLs for offline final-test generation without changing downloads."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid benchmark YAML: {path}")
    return data


def save(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def configure(config_dir: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    changes: dict[str, dict] = {}

    dancetrack_path = config_dir / "dancetrack.yaml"
    dancetrack = load(dancetrack_path)
    shutil.copy2(dancetrack_path, backup_dir / "dancetrack.yaml.before_test")
    dataset = dancetrack.setdefault("dataset", {})
    dataset.setdefault("splits", {})["test"] = "test"
    no_gt = list(dataset.get("no_gt_splits") or [])
    if "test" not in no_gt:
        no_gt.append("test")
    dataset["no_gt_splits"] = no_gt
    save(dancetrack_path, dancetrack)
    changes["dancetrack"] = {
        "test_split": dataset["splits"]["test"],
        "no_gt_splits": dataset["no_gt_splits"],
        "detector_model": dancetrack["detector"]["model"],
    }

    mot20_path = config_dir / "mot20.yaml"
    mot20 = load(mot20_path)
    shutil.copy2(mot20_path, backup_dir / "mot20.yaml.before_test")
    detector = mot20.setdefault("detector", {})
    by_split = detector.setdefault("by_split", {})
    by_split["test"] = {
        "id": "yolox_x_mot20_test",
        "model": "models/yolox_x_MOT20_test.pt",
        "url": "https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT20_test.pt",
    }
    save(mot20_path, mot20)
    changes["mot20"] = {"test_detector": by_split["test"]}

    mot17 = load(config_dir / "mot17.yaml")
    mot17_test = mot17.get("detector", {}).get("by_split", {}).get("test", {})
    expected_mot17 = "models/yolox_x_MOT17_test.pt"
    if mot17_test.get("model") != expected_mot17:
        raise RuntimeError(
            f"MOT17 test detector is {mot17_test.get('model')!r}, expected {expected_mot17!r}"
        )
    changes["mot17"] = {"test_detector": mot17_test}

    manifest = backup_dir / "test_benchmark_configuration.json"
    manifest.write_text(json.dumps(changes, indent=2), encoding="utf-8")
    print(json.dumps(changes, indent=2))
    print(f"Wrote {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir", type=Path, default=Path("boxmot/configs/benchmarks")
    )
    parser.add_argument(
        "--backup-dir", type=Path, default=Path("runs/patch_backups/final_test_configs")
    )
    args = parser.parse_args()
    configure(args.config_dir, args.backup_dir)


if __name__ == "__main__":
    main()

