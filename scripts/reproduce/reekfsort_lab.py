#!/usr/bin/env python3
"""One-command orchestration for ReEKF-SORT server experiments.

The script is restartable: successful tasks are recorded under
``runs/reekfsort_pipeline/state.json`` and skipped on the next invocation.
Use ``--force`` to rerun completed tasks and ``--dry-run`` to inspect commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BENCHMARKS = {
    "dancetrack": {"split": "val", "detector": "yolox_x_dancetrack"},
    "mot17": {"split": "ablation", "detector": "yolox_x_mot17_ablation"},
    "mot20": {"split": "ablation", "detector": "yolox_x_mot20_ablation"},
}
CORE_ROWS = [
    "reekfsort_motion_only",
    "reekfsort_virtual_only",
    "reekfsort_conf_only",
    "reekfsort_full",
    "reekfsort_with_angle",
]
GENERALIZATION_ROWS = [
    "reekfsort_motion_only",
    "reekfsort_virtual_only",
    "reekfsort_full",
    "reekfsort_with_angle",
]
SWEEPS = ("confidence_mode", "lambda_conf", "virtual_interval", "virtual_noise")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PIPELINE_DIR = ROOT / "runs" / "reekfsort_pipeline"
STATE_PATH = PIPELINE_DIR / "state.json"
LOG_DIR = PIPELINE_DIR / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_text(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"schema_version": 1, "tasks": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state: dict) -> None:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


def stream_command(cmd: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(cmd)}\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_task(
    name: str,
    cmd: list[str],
    *,
    args: argparse.Namespace,
    state: dict,
) -> None:
    if not args.force and state["tasks"].get(name, {}).get("status") == "completed":
        print(f"[skip] {name} (already completed; use --force to rerun)")
        return
    print(f"[task] {name}\n  {command_text(cmd)}")
    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    started = time.monotonic()
    task_state = {
        "status": "running",
        "started_at": utc_now(),
        "command": cmd,
        "gpu": args.gpu,
    }
    state["tasks"][name] = task_state
    write_state(state)
    try:
        stream_command(cmd, env=env, log_path=LOG_DIR / f"{name}.log")
    except Exception:
        task_state["status"] = "failed"
        task_state["finished_at"] = utc_now()
        task_state["duration_seconds"] = round(time.monotonic() - started, 3)
        write_state(state)
        raise
    task_state["status"] = "completed"
    task_state["finished_at"] = utc_now()
    task_state["duration_seconds"] = round(time.monotonic() - started, 3)
    write_state(state)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def doctor(args: argparse.Namespace) -> None:
    if args.dry_run:
        print("[doctor] would verify CUDA, GPU, dependencies, disk space, and write environment.json")
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run: bash scripts/reproduce/run_reekfsort_lab.sh setup"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda.is_available() is False. Check NVIDIA driver and install a CUDA PyTorch wheel."
        )
    device_name = torch.cuda.get_device_name(0)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 30 * 1024**3:
        raise SystemExit(
            f"Only {free_bytes / 1024**3:.1f} GiB free. At least 30 GiB is required; "
            "150 GiB is recommended for datasets, caches, and all runs."
        )
    if free_bytes < 80 * 1024**3:
        print(
            f"WARNING: only {free_bytes / 1024**3:.1f} GiB is free. The run may fill the disk; "
            "150 GiB is recommended."
        )

    weights = {}
    for path in sorted((ROOT / "models").glob("yolox_x_*.pt")) if (ROOT / "models").exists() else []:
        weights[path.relative_to(ROOT).as_posix()] = sha256sum(path)
    info = {
        "created_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": device_name,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": capture(["nvidia-smi"]),
        "pip_freeze": capture([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "git_revision": capture(["git", "rev-parse", "HEAD"]),
        "free_disk_gib": round(free_bytes / 1024**3, 2),
        "weight_sha256": weights,
    }
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    (PIPELINE_DIR / "environment.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"GPU ready: {device_name}; PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
    print(f"Wrote {PIPELINE_DIR / 'environment.json'}")


def asset_tasks(args: argparse.Namespace, state: dict) -> None:
    if not args.accept_dataset_terms:
        raise SystemExit(
            "Dataset download requires --accept-dataset-terms after reviewing "
            "docs/trackers/reekfsort_server_setup.md."
        )
    cmd = [
        sys.executable,
        "scripts/reproduce/prepare_reekfsort_assets.py",
        "--accept-dataset-terms",
    ]
    if args.force:
        cmd.append("--overwrite")
    run_task("assets_prepare_and_verify", cmd, args=args, state=state)


def cache_tasks(args: argparse.Namespace, state: dict) -> None:
    for benchmark in args.benchmarks:
        spec = BENCHMARKS[benchmark]
        cmd = [
            sys.executable,
            "-m",
            "boxmot.engine.cli",
            "generate",
            "--benchmark",
            benchmark,
            "--split",
            spec["split"],
            "--detector",
            spec["detector"],
            "--device",
            args.gpu,
        ]
        run_task(f"cache_{benchmark}_{spec['split']}", cmd, args=args, state=state)


def baseline_tasks(args: argparse.Namespace, state: dict) -> None:
    for benchmark in args.benchmarks:
        spec = BENCHMARKS[benchmark]
        for tracker in ("ocsort", "reekfsort"):
            name = f"{benchmark}_{spec['split']}_{tracker}_baseline"
            cmd = [
                sys.executable,
                "-m",
                "boxmot.engine.cli",
                "eval",
                "--benchmark",
                benchmark,
                "--split",
                spec["split"],
                "--tracker",
                tracker,
                "--tracker-backend",
                "python",
                "--project",
                "runs/reekfsort_baselines",
                "--name",
                name,
                "--exist-ok",
            ]
            run_task(f"baseline_{name}", cmd, args=args, state=state)


def experiment_tasks(args: argparse.Namespace, state: dict) -> None:
    if "dancetrack" in args.benchmarks:
        run_task(
            "ablation_dancetrack_val",
            [
                sys.executable,
                "scripts/reproduce/reekfsort_ablation.py",
                "--benchmark",
                "dancetrack",
                "--split",
                "val",
                "--only",
                *CORE_ROWS,
            ],
            args=args,
            state=state,
        )
        for sweep in SWEEPS:
            run_task(
                f"sweep_dancetrack_{sweep}",
                [
                    sys.executable,
                    "scripts/reproduce/reekfsort_sweeps.py",
                    "--benchmark",
                    "dancetrack",
                    "--split",
                    "val",
                    "--sweep",
                    sweep,
                ],
                args=args,
                state=state,
            )

    for benchmark in ("mot17", "mot20"):
        if benchmark not in args.benchmarks:
            continue
        run_task(
            f"ablation_{benchmark}_ablation",
            [
                sys.executable,
                "scripts/reproduce/reekfsort_ablation.py",
                "--benchmark",
                benchmark,
                "--split",
                "ablation",
                "--only",
                *GENERALIZATION_ROWS,
            ],
            args=args,
            state=state,
        )

    run_task(
        "efficiency_ocsort_vs_reekfsort",
        [
            sys.executable,
            "-m",
            "tests.performance.benchmark_fps",
            "--trackers",
            "ocsort,reekfsort",
            "--backends",
            "python",
            "--counts",
            "20,50,100,200",
            "--warmup",
            "200",
            "--frames",
            "1000",
            "--json",
            "runs/reekfsort_pipeline/efficiency.json",
            "--csv",
            "runs/reekfsort_pipeline/efficiency.csv",
        ],
        args=args,
        state=state,
    )


def efficiency_task(args: argparse.Namespace, state: dict) -> None:
    run_task(
        "efficiency_ocsort_vs_reekfsort",
        [
            sys.executable,
            "-m",
            "tests.performance.benchmark_fps",
            "--trackers",
            "ocsort,reekfsort",
            "--backends",
            "python",
            "--counts",
            "20,50,100,200",
            "--warmup",
            "200",
            "--frames",
            "1000",
            "--json",
            "runs/reekfsort_pipeline/efficiency.json",
            "--csv",
            "runs/reekfsort_pipeline/efficiency.csv",
        ],
        args=args,
        state=state,
    )


def smoke_tasks(args: argparse.Namespace, state: dict) -> None:
    run_task(
        "unit_reekfsort",
        [sys.executable, "-m", "pytest", "tests/unit/test_reekfsort.py", "-q"],
        args=args,
        state=state,
    )
    run_task(
        "cli_help",
        [sys.executable, "-m", "boxmot.engine.cli", "--help"],
        args=args,
        state=state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["doctor", "assets", "cache", "smoke", "baselines", "efficiency"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA device index, default: 0.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARKS),
        default=list(BENCHMARKS),
    )
    parser.add_argument("--accept-dataset-terms", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed tasks and overwrite assets.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    state = load_state()
    if args.stage == "doctor":
        doctor(args)
    if args.stage == "assets":
        asset_tasks(args, state)
    if args.stage == "smoke":
        smoke_tasks(args, state)
    if args.stage == "cache":
        cache_tasks(args, state)
    if args.stage == "baselines":
        baseline_tasks(args, state)
    if args.stage == "efficiency":
        efficiency_task(args, state)
    print(f"Pipeline stage '{args.stage}' finished. State: {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

