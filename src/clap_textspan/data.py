"""Versioned waveform length policy; upstream model code is never patched."""

import numpy as np

DURATION_POLICY = "duration_round_half_up_v1"
LEGACY_POLICY = "upstream_librosa_ceil_v1"


def resampled_length(frames, source_rate, target_rate=48000):
    if frames <= 0 or source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio must be nonempty with positive integer sample rates")
    # Exact rational duration, rounded to nearest sample; half samples round up.
    return (2 * int(frames) * int(target_rate) + int(source_rate)) // (
        2 * int(source_rate)
    )


def load_waveform(path, policy=DURATION_POLICY):
    import librosa
    import soundfile as sf

    info = sf.info(path)
    if info.frames <= 0:
        raise ValueError("Empty audio is unsupported")
    if policy == LEGACY_POLICY:
        waveform, _ = librosa.load(path, sr=48000, mono=True)
    elif policy == DURATION_POLICY:
        native, rate = librosa.load(path, sr=None, mono=True)
        expected = resampled_length(len(native), rate)
        if expected == 0:
            raise ValueError("Audio duration is shorter than half a target sample")
        waveform = librosa.resample(
            native,
            orig_sr=rate,
            target_sr=48000,
            res_type="soxr_hq",
            fix=False,
            scale=False,
        )
        waveform = librosa.util.fix_length(waveform, size=expected)
    else:
        raise ValueError(f"Unsupported preprocessing policy: {policy}")
    if waveform.dtype != np.float32 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite float32 decoded audio")
    return waveform, dict(
        policy=policy,
        original_frames=info.frames,
        original_sample_rate=info.samplerate,
        original_channels=info.channels,
        resampled_samples=len(waveform),
        resampler="soxr_hq",
        sample_rate=48000,
    )


from pathlib import Path
import csv
import hashlib
import json
import subprocess
from . import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rows(partition):
    with (ROOT / f"assets/{partition}.csv").open(newline="") as stream:
        return [dict(r, target=int(r["target"])) for r in csv.DictReader(stream)]


def cohort():
    fit, evaluation = rows("fit"), rows("eval")
    sources = {r["source_id"] for r in fit}
    keep = np.array([r["source_id"] not in sources for r in evaluation])
    if len(fit) != 5929 or len(evaluation) != 8732 or keep.sum() != 7962:
        raise ValueError("Frozen cohort changed")
    return fit, evaluation, keep, np.array([r["target"] for r in evaluation])


def fetch(url, path, digest, algorithm="sha256"):
    def checksum(candidate):
        digest = hashlib.new(algorithm)
        with Path(candidate).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "4",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        if checksum(temporary) != digest:
            raise ValueError(f"Download checksum mismatch: {path}")
        temporary.replace(path)
    if checksum(path) != digest:
        raise ValueError(f"Checksum mismatch: {path}")


def setup(directory):
    """Download checkpoint/tokenizer and install only the pinned upstream source."""
    import sys, tarfile

    directory = Path(directory).resolve()
    pins = read_json(ROOT / "assets/protocol.json")
    fetch(
        pins["checkpoint_url"],
        directory / "630k-audioset-best.pt",
        pins["checkpoint_sha256"],
    )
    text = directory / "roberta-base"
    fetch(
        pins["roberta_config_url"], text / "config.json", pins["roberta_config_sha256"]
    )
    for name, digest in pins["tokenizer_files"].items():
        fetch(
            f"https://huggingface.co/FacebookAI/roberta-base/resolve/{pins['roberta_revision']}/{name}",
            text / name,
            digest,
        )
    archive = directory / "clap.tar.gz"
    fetch(
        f"https://codeload.github.com/LAION-AI/CLAP/tar.gz/{pins['clap_commit']}",
        archive,
        pins["clap_archive_sha256"],
    )
    upstream = directory / f"CLAP-{pins['clap_commit']}"
    if not upstream.exists():
        with tarfile.open(archive) as source:
            source.extractall(directory, filter="data")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(upstream),
        ],
        check=True,
    )


def verify_data(directory):
    fit, evaluation, keep, _ = cohort()
    base = Path(directory).resolve()
    for records in [fit, evaluation]:
        for i, row in enumerate(records):
            path = base / row["path"]
            if not path.is_relative_to(base) or sha256(path) != row["sha256"]:
                raise ValueError(f"Missing or changed audio: {path}")
            if i % 1000 == 0:
                print(f"Verified {i}/{len(records)}", flush=True)
    print(
        f"Verified {len(fit)} fit, {len(evaluation)} evaluation, {keep.sum()} source-disjoint clips"
    )


def write_csv(path, records):
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def download_data(directory):
    """Official archives only. Manifests verify the exact decoded WAV files afterwards."""
    import tarfile
    import py7zr

    directory = Path(directory).resolve()
    pins = read_json(ROOT / "assets/datasets.json")
    for split, digest in pins["clotho"]["archives"].items():
        name = f"clotho_audio_{split}.7z"
        archive = directory / "data/clotho" / name
        destination = directory / "data/clotho" / split
        if not destination.exists():
            fetch(
                pins["clotho"]["url"] + name + "?download=1",
                archive,
                digest,
                algorithm="md5",
            )
            with py7zr.SevenZipFile(archive) as source:
                source.extractall(destination)
    urban = pins["urbansound8k"]
    archive = directory / "data/urbansound8k/UrbanSound8K.tar.gz"
    if not (archive.parent / "UrbanSound8K").exists():
        fetch(urban["url"] + "?download=1", archive, urban["md5"], algorithm="md5")
        with tarfile.open(archive) as source:
            source.extractall(archive.parent, filter="data")
    verify_data(directory)
