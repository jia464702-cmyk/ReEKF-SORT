#!/usr/bin/env python3
"""Download the datasets and detector weights used by ReEKF-SORT experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from tqdm import tqdm


@dataclass(frozen=True)
class DetectorAsset:
    benchmark: str
    split: str
    detector_id: str
    model: str
    url: str
    imgsz: tuple[int, int]
    conf: float


@dataclass(frozen=True)
class DatasetAsset:
    benchmark: str
    split: str
    root: str
    source: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_OFFICIAL_SOURCES = {
    "dancetrack": (
        "[Project](https://dancetrack.github.io/), "
        "[GitHub](https://github.com/DanceTrack/DanceTrack), "
        "[Hugging Face mirror](https://huggingface.co/datasets/noahcao/dancetrack)"
    ),
    "mot17": "[MOTChallenge/Codabench MOT17](https://www.codabench.org/competitions/10049/)",
    "mot20": "[MOTChallenge/Codabench MOT20](https://www.codabench.org/competitions/10050/)",
}

DATASET_NOTES = {
    "dancetrack": "DanceTrack is for non-commercial research purposes only; do not redistribute it in this release.",
    "mot17": "Download from official challenge pages or documented mirrors; do not bundle data.",
    "mot20": "Download from official challenge pages or documented mirrors; do not bundle data.",
}

WEIGHT_SHA256 = {
    "models/yolox_x_dancetrack.pt": "f76e036f872a57710d9aebdfad2730e1c0e78bc3805e0526fa4b04a9e3c1d13d",
    "models/yolox_x_MOT17_ablation.pt": "26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a",
    "models/yolox_x_MOT20_ablation.pt": "c8a49d9a58ab6dbc59e2f5daebb552513f6f5474acd9f62b47b4ef738e4900a3",
}

WEIGHT_PROVENANCE = (
    "Third-party/release reproduction YOLOX checkpoint; not bundled and not claimed as an official "
    "Megvii YOLOX checkpoint."
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def benchmark_cfg(name: str) -> dict:
    return load_yaml(repo_root() / "boxmot" / "configs" / "benchmarks" / f"{name}.yaml")


def split_value(value, split: str):
    if isinstance(value, dict):
        return value.get(split) or value.get("default") or next(iter(value.values()))
    return value


def detector_asset(name: str, split: str) -> DetectorAsset:
    cfg = benchmark_cfg(name)
    det = dict(cfg["detector"])
    override = (det.get("by_split") or {}).get(split)
    if override:
        det.update(override)
    imgsz = tuple(int(v) for v in det.get("imgsz", [800, 1440]))
    return DetectorAsset(
        benchmark=name,
        split=split,
        detector_id=str(det["id"]),
        model=str(det["model"]),
        url=str(det["url"]),
        imgsz=(imgsz[0], imgsz[1]),
        conf=float(det.get("conf", 0.01)),
    )


def dataset_asset(name: str, split: str) -> DatasetAsset:
    cfg = benchmark_cfg(name)
    download = cfg.get("download") or {}
    dataset = cfg["dataset"]
    source = split_value(download.get("dataset"), split)
    return DatasetAsset(
        benchmark=name,
        split=split,
        root=str(dataset["root"]),
        source=str(source),
    )


def download_http(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        with tmp.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def extract_zip(zip_path: Path, dest: Path, overwrite: bool = False) -> None:
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)


def normalize_dataset_root(dest: Path) -> None:
    nested = dest / dest.name
    if not nested.is_dir():
        pass
    else:
        for child in nested.iterdir():
            target = dest / child.name
            if target.exists():
                continue
            child.replace(target)
        try:
            nested.rmdir()
        except OSError:
            pass

    if dest.name == "test1" and not (dest / "val").exists():
        seq_dirs = [p for p in dest.iterdir() if p.is_dir() and p.name.startswith("dancetrack")]
        if seq_dirs:
            val_dir = dest / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            for seq_dir in seq_dirs:
                seq_dir.replace(val_dir / seq_dir.name)


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for hf:// dataset sources. "
            "Install it with: python -m pip install huggingface_hub"
        ) from exc


def download_hf_folder(source: str, dest: Path, overwrite: bool = False) -> None:
    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    rel = source[len("hf://") :]
    parts = rel.split("/")
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
    allow_patterns = [f"{subfolder}/**"] if subfolder else None
    dest.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=dest,
        local_dir_use_symlinks=False,
        force_download=overwrite,
    )
    if subfolder:
        src = Path(local_dir) / subfolder
        if src.exists() and src != dest:
            for child in src.iterdir():
                target = dest / child.name
                if target.exists():
                    continue
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)


def download_dataset(asset: DatasetAsset, overwrite: bool = False) -> Path:
    dest = repo_root() / asset.root
    if asset.benchmark == "mot17":
        from boxmot.utils.mot17_parquet import setup_mot17_from_parquet

        cache = dest / ".parquet_cache"
        for repo_path in (
            "data/seqinfo/seqinfo.parquet",
            "data/gt/train-00000-of-00001.parquet",
            "data/detections/frcnn/train-00000-of-00001.parquet",
        ):
            local_path = cache / repo_path
            url = f"https://huggingface.co/datasets/Lekim89/mot17-parquet/resolve/main/{repo_path}"
            download_http(url, local_path, overwrite=overwrite)
        setup_mot17_from_parquet(dest=dest, split=asset.split, detector="FRCNN", overwrite=overwrite)
        return dest

    if asset.source.startswith("http"):
        filename = Path(urlparse(asset.source).path).name or f"{asset.benchmark}.zip"
        zip_path = repo_root() / "downloads" / filename
        download_http(asset.source, zip_path, overwrite=overwrite)
        extract_zip(zip_path, dest.parent, overwrite=overwrite)
        normalize_dataset_root(dest)
    elif asset.source.startswith("hf://"):
        from boxmot.utils.download import download_hf_dataset_subfolder

        rel = asset.source[len("hf://") :]
        parts = rel.split("/")
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        download_hf_dataset_subfolder(repo_id, subfolder, dest, overwrite=overwrite)
    else:
        raise ValueError(f"Unsupported dataset source for {asset.benchmark}: {asset.source}")
    return dest


def dataset_status(asset: DatasetAsset, local_path: Path) -> str:
    if asset.benchmark == "dancetrack":
        return "present" if (local_path / asset.split).is_dir() else "missing"
    if asset.benchmark == "mot17":
        marker = local_path / asset.split / ".parquet_setup_complete"
        seq_dirs = [p for p in (local_path / asset.split).glob("MOT17-*") if p.is_dir()]
        has_eval_files = any((p / "gt" / "gt.txt").exists() and (p / "det" / "det.txt").exists() for p in seq_dirs)
        return "present" if marker.exists() and seq_dirs and has_eval_files else "incomplete"
    if asset.benchmark == "mot20":
        return "present" if (local_path / asset.split).is_dir() else "missing"
    return "present" if local_path.exists() else "missing"


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def write_versions(
    datasets: list[DatasetAsset],
    detectors: list[DetectorAsset],
    dataset_paths: dict[str, Path],
    detector_paths: dict[str, Path],
) -> Path:
    target = repo_root() / "docs" / "trackers" / "reekfsort_assets.md"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# ReEKF-SORT Data And Weight Versions",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "The open-source release package does not include datasets, detector weights, run outputs, or caches.",
        "",
        "## Datasets",
        "",
        "| Benchmark | Split | Local path | Official or stable source | Helper download source | Redistribution note | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for asset in datasets:
        local_path = dataset_paths[asset.benchmark]
        status = dataset_status(asset, local_path)
        lines.append(
            f"| `{asset.benchmark}` | `{asset.split}` | `{relpath(local_path)}` | "
            f"{DATASET_OFFICIAL_SOURCES.get(asset.benchmark, '')} | `{asset.source}` | "
            f"{DATASET_NOTES.get(asset.benchmark, '')} | `{status}` |"
        )
    lines.extend(
        [
            "",
            "## Detector Weights",
            "",
            "| Benchmark | Split | Detector | Local path | Source URL | SHA256 | Provenance | Image size | Confidence | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for asset in detectors:
        local_path = detector_paths[asset.benchmark]
        status = "present" if local_path.exists() else "missing"
        lines.append(
            "| "
            f"`{asset.benchmark}` | `{asset.split}` | `{asset.detector_id}` | "
            f"`{relpath(local_path)}` | `{asset.url}` | `{WEIGHT_SHA256.get(asset.model, 'TO_BE_CONFIRMED')}` | "
            f"{WEIGHT_PROVENANCE} | `{asset.imgsz}` | `{asset.conf}` | `{status}` |"
        )
    lines.extend(
        [
            "",
            "If these URLs are not the final stable hosting location, publish the weights to a controlled Hugging Face repository, GitHub Release, or Zenodo record and update both the URLs and SHA256 values.",
            "",
            "## ReID Weights",
            "",
            "ReEKF-SORT is a motion-only tracker. No ReID weights are required for the reported ReEKF-SORT experiments.",
            "",
            "## Commands",
            "",
            "```bash",
            "python scripts/reproduce/prepare_reekfsort_assets.py",
            "python scripts/reproduce/reekfsort_ablation.py --benchmark dancetrack --split val",
            "python scripts/reproduce/reekfsort_ablation.py --benchmark mot17 --split ablation",
            "python scripts/reproduce/reekfsort_ablation.py --benchmark mot20 --split ablation",
            "```",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_at": generated_at,
        "datasets": [asset.__dict__ | {"local_path": relpath(dataset_paths[asset.benchmark])} for asset in datasets],
        "detectors": [asset.__dict__ | {"local_path": relpath(detector_paths[asset.benchmark])} for asset in detectors],
        "reid": "not required for ReEKF-SORT",
    }
    (repo_root() / "scripts" / "reproduce" / "reekfsort_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-datasets", action="store_true", help="Only download detector weights.")
    parser.add_argument("--skip-weights", action="store_true", help="Only download datasets.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["dancetrack", "mot17", "mot20"],
        help="Prepare selected benchmarks only. Defaults to all three.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = [("dancetrack", "val"), ("mot17", "ablation"), ("mot20", "ablation")]
    if args.only:
        selected = set(args.only)
        plan = [(name, split) for name, split in plan if name in selected]
    datasets = [dataset_asset(name, split) for name, split in plan]
    detectors = [detector_asset(name, split) for name, split in plan]

    dataset_paths: dict[str, Path] = {}
    detector_paths: dict[str, Path] = {}

    if not args.skip_datasets:
        for asset in datasets:
            dataset_paths[asset.benchmark] = download_dataset(asset, overwrite=args.overwrite)
    else:
        for asset in datasets:
            dataset_paths[asset.benchmark] = repo_root() / asset.root

    if not args.skip_weights:
        for asset in detectors:
            target = repo_root() / asset.model
            detector_paths[asset.benchmark] = download_http(asset.url, target, overwrite=args.overwrite)
    else:
        for asset in detectors:
            detector_paths[asset.benchmark] = repo_root() / asset.model

    version_doc = write_versions(datasets, detectors, dataset_paths, detector_paths)
    print(f"Wrote {version_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
