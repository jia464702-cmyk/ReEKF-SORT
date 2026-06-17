#!/usr/bin/env python3
"""Run the ReEKF-SORT ablation matrix from cached benchmark detections.

This script intentionally edits only the default values in
boxmot/configs/trackers/reekfsort.yaml, runs one evaluation, and then restores
the original file at the end. Generate detections before running ablations:

    boxmot generate --benchmark dancetrack --split val --detector yolox_x_dancetrack
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ABLATIONS = {
    "reekfsort_motion_only": {
        "use_virtual_observation": False,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_virtual_only": {
        "use_virtual_observation": True,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_conf_only": {
        "use_virtual_observation": False,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_angle_only": {
        "use_virtual_observation": False,
        "use_confidence_cost": False,
        "use_angle_cost": True,
    },
    "reekfsort_no_virtual": {
        "use_virtual_observation": False,
        "use_confidence_cost": True,
        "use_angle_cost": True,
    },
    "reekfsort_no_conf": {
        "use_virtual_observation": True,
        "use_confidence_cost": False,
        "use_angle_cost": True,
    },
    "reekfsort_no_angle": {
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_full": {
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": True,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracker_config_path() -> Path:
    return repo_root() / "boxmot" / "configs" / "trackers" / "reekfsort.yaml"


def set_ablation_defaults(config_path: Path, switches: dict[str, bool]) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key, value in switches.items():
        if key not in cfg:
            raise KeyError(f"Missing ReEKF-SORT ablation key in config: {key}")
        cfg[key]["default"] = bool(value)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_eval(args: argparse.Namespace, ablation_name: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "boxmot.engine.cli",
        "eval",
        "--benchmark",
        args.benchmark,
        "--split",
        args.split,
        "--tracker",
        "reekfsort",
        "--tracker-backend",
        "python",
        "--project",
        args.project,
        "--name",
        ablation_name,
        "--exist-ok",
    ]
    if args.detection_source:
        cmd.extend(["--detection-source", args.detection_source])
    if args.detector:
        cmd.extend(["--detector", args.detector])
    if args.dry_run:
        print(" ".join(cmd))
        return
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(cmd, cwd=repo_root(), check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Benchmark id, e.g. dancetrack, mot17, mot20.")
    parser.add_argument("--split", required=True, help="Dataset split, e.g. val or ablation.")
    parser.add_argument("--detector", default=None, help="Optional detector profile or checkpoint override.")
    parser.add_argument(
        "--detection-source",
        choices=["public", "private"],
        default=None,
        help="Use public detections when reproducing MOTChallenge public-detector results.",
    )
    parser.add_argument("--project", default="runs/reekfsort_ablation", help="Output project directory.")
    parser.add_argument("--only", nargs="*", choices=sorted(ABLATIONS), help="Run selected ablations only.")
    parser.add_argument("--dry-run", action="store_true", help="Print eval commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = tracker_config_path()
    original = config_path.read_text(encoding="utf-8")
    selected = args.only or list(ABLATIONS)

    try:
        for ablation_name in selected:
            set_ablation_defaults(config_path, ABLATIONS[ablation_name])
            run_eval(args, ablation_name)
    finally:
        config_path.write_text(original, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
