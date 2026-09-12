#!/usr/bin/env python3
"""Build MOT17/MOT20 ByteTrack val-half splits from official ZIP files.

This offline path is intended for machines that cannot reach Hugging Face.  It
extracts only the frames needed by the ablation split, so the full train/test
archives do not have to be unpacked on disk.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_SEQUENCES = {
    "mot17": {
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    },
    "mot20": {"MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_names(archive: zipfile.ZipFile, benchmark: str) -> set[str]:
    prefix = benchmark.upper()
    names: set[str] = set()
    for member in archive.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[0] == prefix and parts[1] == "train":
            sequence = parts[2]
            if benchmark == "mot17" and not sequence.endswith("-FRCNN"):
                continue
            names.add(sequence)
    return names


def _read_seqinfo(archive: zipfile.ZipFile, member: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read_string(archive.read(member).decode("utf-8-sig"))
    if "Sequence" not in parser or "seqLength" not in parser["Sequence"]:
        raise RuntimeError(f"Invalid seqinfo.ini: {member}")
    return parser


def _write_seqinfo(path: Path, parser: configparser.ConfigParser, sequence: str, length: int) -> None:
    section = parser["Sequence"]
    section["name"] = sequence
    section["imDir"] = "img1"
    section["seqLength"] = str(length)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        parser.write(handle, space_around_delimiters=False)


def _filter_annotation(raw: bytes, start_frame: int, output: Path) -> int:
    rows = []
    text = raw.decode("utf-8-sig").splitlines()
    for row in csv.reader(text):
        if not row:
            continue
        frame = int(float(row[0]))
        if frame < start_frame:
            continue
        row[0] = str(frame - start_frame + 1)
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return len(rows)


def _build_one(archive: zipfile.ZipFile, benchmark: str, sequence: str, output: Path) -> dict:
    prefix = f"{benchmark.upper()}/train/{sequence}"
    parser = _read_seqinfo(archive, f"{prefix}/seqinfo.ini")
    total_length = int(parser["Sequence"]["seqLength"])
    split_frame = total_length // 2 + 1
    start_frame = split_frame + 1
    split_length = total_length - split_frame
    sequence_dir = output / sequence
    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True)

    extracted = 0
    for original_frame in range(start_frame, total_length + 1):
        source = f"{prefix}/img1/{original_frame:06d}.jpg"
        target = image_dir / f"{original_frame - start_frame + 1:06d}.jpg"
        try:
            with archive.open(source) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        except KeyError as exc:
            raise RuntimeError(f"Archive is missing required frame: {source}") from exc
        extracted += 1

    if extracted != split_length:
        raise RuntimeError(f"{sequence}: expected {split_length} images, extracted {extracted}")

    _write_seqinfo(sequence_dir / "seqinfo.ini", parser, sequence, split_length)
    counts = {"images": extracted}
    for folder, filename in (("gt", "gt.txt"), ("det", "det.txt")):
        member = f"{prefix}/{folder}/{filename}"
        try:
            raw = archive.read(member)
        except KeyError:
            if folder == "gt":
                raise RuntimeError(f"Archive is missing required annotation: {member}")
            continue
        counts[folder] = _filter_annotation(raw, start_frame, sequence_dir / folder / filename)

    return {
        "sequence": sequence,
        "source_seq_length": total_length,
        "source_start_frame": start_frame,
        "ablation_seq_length": split_length,
        **counts,
    }


def build(benchmark: str, archive_path: Path, data_root: Path) -> Path:
    expected = EXPECTED_SEQUENCES[benchmark]
    destination = data_root / benchmark.upper() / "ablation"
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"Destination is not empty: {destination}. Preserve it and inspect before retrying."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        actual = _sequence_names(archive, benchmark)
        if actual != expected:
            raise RuntimeError(
                f"Unexpected {benchmark.upper()} train sequences. "
                f"Missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        with tempfile.TemporaryDirectory(prefix=f".{benchmark}_ablation_", dir=destination.parent) as temp:
            staging = Path(temp) / "ablation"
            staging.mkdir()
            records = []
            for index, sequence in enumerate(sorted(expected), start=1):
                print(
                    f"[{index}/{len(expected)}] Extracting {sequence} val-half...",
                    flush=True,
                )
                record = _build_one(archive, benchmark, sequence, staging)
                records.append(record)
                print(
                    f"[{index}/{len(expected)}] Completed {sequence}: "
                    f"{record['images']} images, {record.get('gt', 0)} GT rows",
                    flush=True,
                )
            manifest = {
                "benchmark": benchmark,
                "protocol": "ByteTrack val-half: keep original frames > N//2+1 and renumber from 1",
                "source_archive": str(archive_path.resolve()),
                "source_sha256": _sha256(archive_path),
                "sequences": records,
            }
            (staging / "offline_ablation_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            if destination.exists():
                destination.rmdir()
            staging.replace(destination)

    print(f"Prepared {benchmark.upper()} ablation split: {destination}")
    for record in records:
        print(
            f"  {record['sequence']}: {record['images']} images, "
            f"{record.get('gt', 0)} GT rows, frames {record['source_start_frame']}-"
            f"{record['source_seq_length']}"
        )
    return destination


def verify(benchmark: str, data_root: Path) -> None:
    root = data_root / benchmark.upper() / "ablation"
    found = {path.name for path in root.iterdir() if path.is_dir()}
    expected = EXPECTED_SEQUENCES[benchmark]
    if found != expected:
        raise RuntimeError(f"Sequence mismatch in {root}: found {sorted(found)}")
    for sequence in sorted(expected):
        sequence_dir = root / sequence
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(sequence_dir / "seqinfo.ini")
        length = int(parser["Sequence"]["seqLength"])
        images = sorted((sequence_dir / "img1").glob("*.jpg"))
        if len(images) != length or images[0].name != "000001.jpg" or images[-1].name != f"{length:06d}.jpg":
            raise RuntimeError(f"Invalid image sequence: {sequence}")
        for annotation in (sequence_dir / "gt" / "gt.txt", sequence_dir / "det" / "det.txt"):
            if not annotation.exists():
                continue
            with annotation.open(encoding="utf-8") as handle:
                frames = [int(float(row[0])) for row in csv.reader(handle) if row]
            if frames and (min(frames) < 1 or max(frames) > length):
                raise RuntimeError(f"Out-of-range annotation frame in {annotation}")
    print(f"Verified {benchmark.upper()}: {len(expected)} sequences at {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("mot17", "mot20", "both"), default="both")
    parser.add_argument("--mot17-zip", type=Path, default=Path("downloads/MOT17.zip"))
    parser.add_argument("--mot20-zip", type=Path, default=Path("downloads/MOT20.zip"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("boxmot/engine/eval/trackeval/data"),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = ("mot17", "mot20") if args.benchmark == "both" else (args.benchmark,)
    for benchmark in benchmarks:
        archive = getattr(args, f"{benchmark}_zip")
        if not args.verify_only:
            if not archive.is_file():
                raise FileNotFoundError(archive)
            build(benchmark, archive, args.data_root)
        verify(benchmark, args.data_root)


if __name__ == "__main__":
    main()

