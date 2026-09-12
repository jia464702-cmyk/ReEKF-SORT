#!/usr/bin/env python3
"""Merge and verify the two official DanceTrack test ZIP archives offline."""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_SEQUENCES = 35
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sequence_length(sequence_dir: Path) -> int:
    parser = configparser.ConfigParser()
    seqinfo = sequence_dir / "seqinfo.ini"
    if not seqinfo.is_file():
        raise RuntimeError(f"Missing {seqinfo}")
    parser.read(seqinfo, encoding="utf-8-sig")
    try:
        return parser.getint("Sequence", "seqLength")
    except (configparser.Error, ValueError) as exc:
        raise RuntimeError(f"Invalid seqLength in {seqinfo}") from exc


def verify(destination: Path) -> list[dict[str, object]]:
    if not destination.is_dir():
        raise RuntimeError(f"Missing DanceTrack test split: {destination}")
    sequences = sorted(path for path in destination.iterdir() if path.is_dir())
    if len(sequences) != EXPECTED_SEQUENCES:
        raise RuntimeError(
            f"Expected {EXPECTED_SEQUENCES} DanceTrack test sequences, found {len(sequences)}"
        )
    report: list[dict[str, object]] = []
    for sequence_dir in sequences:
        image_dir = sequence_dir / "img1"
        if not image_dir.is_dir():
            raise RuntimeError(f"Missing image directory: {image_dir}")
        images = sum(1 for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        expected = sequence_length(sequence_dir)
        if images != expected:
            raise RuntimeError(f"{sequence_dir.name}: seqLength={expected}, images={images}")
        report.append({"sequence": sequence_dir.name, "frames": images})
    print(
        f"Verified DanceTrack test: {len(report)} sequences, "
        f"{sum(int(item['frames']) for item in report)} frames at {destination}"
    )
    for item in report:
        print(f"  {item['sequence']}: {item['frames']} images")
    return report


def extract_archive(archive: Path, top_level: str, staging: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"Official archive not found: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        selected = 0
        for info in bundle.infolist():
            source = PurePosixPath(info.filename)
            if not source.parts or source.parts[0] != top_level or len(source.parts) <= 1:
                continue
            relative = PurePosixPath(*source.parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
            target = staging.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise RuntimeError(f"Duplicate member across test archives: {relative}")
                with bundle.open(info) as source_file, target.open("wb") as output:
                    shutil.copyfileobj(source_file, output, length=1024 * 1024)
            selected += 1
        if selected == 0:
            raise RuntimeError(f"No {top_level}/ files found in {archive}")


def prepare(test1_zip: Path, test2_zip: Path, destination: Path) -> None:
    if destination.exists():
        print(f"Reusing existing DanceTrack test split: {destination}")
        verify(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dancetrack_test_", dir=destination.parent) as temp:
        staging = Path(temp) / "test"
        staging.mkdir()
        extract_archive(test1_zip, "test1", staging)
        extract_archive(test2_zip, "test2", staging)
        report = verify(staging)
        shutil.move(str(staging), str(destination))
        manifest = {
            "benchmark": "dancetrack",
            "split": "test",
            "source_archives": [str(test1_zip), str(test2_zip)],
            "sequence_count": len(report),
            "frame_count": sum(int(item["frames"]) for item in report),
            "sequences": report,
        }
        (destination / "offline_test_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    verify(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test1-zip", type=Path, default=Path("downloads/test1.zip"))
    parser.add_argument("--test2-zip", type=Path, default=Path("downloads/test2.zip"))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("boxmot/engine/eval/trackeval/data/test1/test"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.destination)
    else:
        prepare(args.test1_zip, args.test2_zip, args.destination)


if __name__ == "__main__":
    main()

