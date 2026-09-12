#!/usr/bin/env python3
"""Run the ReEKF-SORT ablation matrix from cached benchmark detections.

This script intentionally edits only the default values in
boxmot/configs/trackers/reekfsort.yaml, runs one evaluation, and then restores
the original file at the end. Generate detections before running ablations:

    boxmot generate --benchmark dancetrack --split val --detector yolox_x_dancetrack
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ABLATIONS = {
    "reekfsort_motion_only": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": False,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_virtual_only": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": True,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_conf_only": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": False,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_angle_only": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": False,
        "use_confidence_cost": False,
        "use_angle_cost": True,
    },
    "reekfsort_no_virtual": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": False,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_no_conf": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": True,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_no_angle": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_full": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_with_angle": {
        "use_observation_centric": False,
        "use_ocr": False,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": True,
        "lambda_angle": 0.2,
    },
    "reekfsort_revised_backbone": {
        "use_observation_centric": True,
        "use_ocr": True,
        "use_virtual_observation": False,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_revised_virtual_only": {
        "use_observation_centric": True,
        "use_ocr": True,
        "use_virtual_observation": True,
        "use_confidence_cost": False,
        "use_angle_cost": False,
    },
    "reekfsort_revised_conf_only": {
        "use_observation_centric": True,
        "use_ocr": True,
        "use_virtual_observation": False,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_revised_full": {
        "use_observation_centric": True,
        "use_ocr": True,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": False,
    },
    "reekfsort_revised_with_angle": {
        "use_observation_centric": True,
        "use_ocr": True,
        "use_virtual_observation": True,
        "use_confidence_cost": True,
        "use_angle_cost": True,
        "lambda_angle": 0.2,
    },
}

PAPER_DEFAULTS = {
    "dancetrack": {"det_thresh": 0.6},
    "mot17": {"det_thresh": 0.6},
    "mot20": {"det_thresh": 0.6},
}

FINAL_METHOD_DEFAULTS = {
    "lambda_conf": 1.2,
    "lambda_angle": 0.0,
    "confidence_cost_mode": "absolute",
    "use_angle_cost": False,
    "virtual_update_interval": 6,
    "virtual_obs_noise_scale": 10.0,
    "delta_t": 3,
    "inertia": 0.1,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracker_config_path() -> Path:
    return repo_root() / "boxmot" / "configs" / "trackers" / "reekfsort.yaml"


def set_ablation_defaults(config_path: Path, switches: dict[str, object]) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key, value in switches.items():
        if key not in cfg:
            raise KeyError(f"Missing ReEKF-SORT ablation key in config: {key}")
        cfg[key]["default"] = value
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def write_run_manifest(
    args: argparse.Namespace, ablation_name: str, cmd: list[str], config_path: Path
) -> None:
    output_dir = repo_root() / args.project
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": ablation_name,
        "benchmark": args.benchmark,
        "split": args.split,
        "detector": args.detector,
        "detection_source": args.detection_source,
        "command": cmd,
        "tracker_config": yaml.safe_load(config_path.read_text(encoding="utf-8")),
    }
    manifest_name = f"{args.benchmark}_{args.split}_{ablation_name}_manifest.json"
    (output_dir / manifest_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_eval(
    args: argparse.Namespace, ablation_name: str, config_path: Path
) -> None:
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
        f"{args.benchmark}_{args.split}_{ablation_name}",
        "--exist-ok",
        "--device",
        args.device,
        "--show-timing",
    ]
    if args.detection_source:
        cmd.extend(["--detection-source", args.detection_source])
    if args.detector:
        cmd.extend(["--detector", args.detector])
    if args.dry_run:
        print(" ".join(cmd))
        return
    write_run_manifest(args, ablation_name, cmd, config_path)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("BOXMOT_NO_REID", "1")
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
    parser.add_argument(
        "--project",
        default="runs",
        help="Cache and output root. Keep this as runs so existing dets_n_embs caches are reused.",
    )
    parser.add_argument("--device", default="0", help="Runtime device used if cache generation is needed.")
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
            defaults = dict(FINAL_METHOD_DEFAULTS)
            defaults.update(PAPER_DEFAULTS.get(args.benchmark.lower(), {}))
            defaults.update(ABLATIONS[ablation_name])
            set_ablation_defaults(config_path, defaults)
            run_eval(args, ablation_name, config_path)
    finally:
        config_path.write_text(original, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
