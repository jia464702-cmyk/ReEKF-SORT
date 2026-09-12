#!/usr/bin/env python3
"""Extract and verify MOT17/MOT20 test splits from the official ZIP files.

The script is deliberately offline: it never downloads data and only extracts
the ``<BENCHMARK>/test`` subtree into the layout expected by BoxMOT.
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_SEQUENCES = {"mot17": 21, "mot20": 4}
DEFAULT_ZIPS = {
    "mot17": Path("downloads/MOT17.zip"),
    "mot20": Path("downloads/MOT20.zip"),
}


def _sequence_length(sequence_dir: Path) -> int:
    config = configparser.ConfigParser()
    seqinfo = sequence_dir / "seqinfo.ini"
    if not seqinfo.is_file():
        raise RuntimeError(f"Missing {seqinfo}")
    config.read(seqinfo, encoding="utf-8-sig")
    try:
        return config.getint("Sequence", "seqLength")
    except (configparser.Error, ValueError) as exc:
        raise RuntimeError(f"Invalid seqLength in {seqinfo}") from exc


def verify(benchmark: str, destination: Path) -> list[dict[str, object]]:
    if not destination.is_dir():
        raise RuntimeError(f"Missing test split: {destination}")
    sequences = sorted(path for path in destination.iterdir() if path.is_dir())
    expected = EXPECTED_SEQUENCES[benchmark]
    if len(sequences) != expected:
        raise RuntimeError(
            f"{benchmark.upper()}: expected {expected} test sequences, found {len(sequences)}"
        )

    report: list[dict[str, object]] = []
    for sequence_dir in sequences:
        image_dir = sequence_dir / "img1"
        if not image_dir.is_dir():
            raise RuntimeError(f"Missing image directory: {image_dir}")
        image_count = sum(
            1 for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        seq_length = _sequence_length(sequence_dir)
        if image_count != seq_length:
            raise RuntimeError(
                f"{sequence_dir.name}: seqLength={seq_length}, images={image_count}"
            )
        report.append(
            {"sequence": sequence_dir.name, "frames": image_count, "seq_length": seq_length}
        )
    print(
        f"Verified {benchmark.upper()} test: {len(report)} sequences, "
        f"{sum(int(item['frames']) for item in report)} frames at {destination}"
    )
    for item in report:
        print(f"  {item['sequence']}: {item['frames']} images")
    return report


def extract(benchmark: str, archive: Path, data_root: Path, force: bool) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"Official archive not found: {archive}")
    destination = data_root / benchmark.upper() / "test"
    if destination.exists() and not force:
        print(f"Reusing existing test split: {destination}")
        verify(benchmark, destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = PurePosixPath(benchmark.upper(), "test")
    with tempfile.TemporaryDirectory(
        prefix=f".{benchmark}_test_", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "test"
        staging.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            members = []
            for info in bundle.infolist():
                source = PurePosixPath(info.filename)
                if source.parts[:2] != prefix.parts or len(source.parts) <= 2:
                    continue
                relative = PurePosixPath(*source.parts[2:])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
                members.append((info, relative))
            if not members:
                raise RuntimeError(f"No {prefix.as_posix()}/ files found in {archive}")
            for info, relative in members:
                target = staging.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        report = verify(benchmark, staging)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staging), str(destination))
        manifest = {
            "benchmark": benchmark,
            "split": "test",
            "source_archive": str(archive),
            "sequence_count": len(report),
            "frame_count": sum(int(item["frames"]) for item in report),
            "sequences": report,
        }
        (destination / "offline_test_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    verify(benchmark, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("mot17", "mot20", "both"), default="both")
    parser.add_argument("--mot17-zip", type=Path, default=DEFAULT_ZIPS["mot17"])
    parser.add_argument("--mot20-zip", type=Path, default=DEFAULT_ZIPS["mot20"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("boxmot/engine/eval/trackeval/data"),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    benchmarks = ("mot17", "mot20") if args.benchmark == "both" else (args.benchmark,)
    archives = {"mot17": args.mot17_zip, "mot20": args.mot20_zip}
    for benchmark in benchmarks:
        destination = args.data_root / benchmark.upper() / "test"
        if args.verify_only:
            verify(benchmark, destination)
        else:
            extract(benchmark, archives[benchmark], args.data_root, args.force)


if __name__ == "__main__":
    main()

