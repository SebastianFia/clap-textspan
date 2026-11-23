"""Paper figures and portable atlas data. No GPU imports for plotting saved CSVs."""

import csv
import json
from pathlib import Path
import numpy as np
from . import ROOT
from .data import read_json, save_json


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def plots(source, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    def save(fig, name):
        fig.tight_layout(pad=0.7)
        for suffix in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    rows = [
        r for r in table(source / "removal.csv") if r["cohort"] == "source-disjoint"
    ]
    x = np.array([int(r["k"]) for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.fill_between(
        x,
        [100 * float(r["random_p05"]) for r in rows],
        [100 * float(r["random_p95"]) for r in rows],
        color="#b5a4d4",
        alpha=0.28,
        lw=0,
    )
    for key, label, color, marker in [
        ("first_to_last", "First to last", "#157c80", "s"),
        ("last_to_first", "Last to first", "#da7541", "^"),
        ("random_mean", "Random order", "#8061a8", "o"),
    ]:
        ax.plot(
            x,
            [100 * float(r[key]) for r in rows],
            label=label,
            color=color,
            marker=marker,
            ms=3,
            lw=1.5,
        )
    ax.set(
        xlabel="Number of blocks mean-ablated (MHA + MLP)",
        ylabel="Top-1 accuracy (%)",
        xticks=range(0, 13, 2),
        ylim=(0, 85),
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.15)
    save(fig, "cumulative-removal-order")
    rows = table(source / "reconstruction.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for key, label, color in [
        ("audio_v2", "LLM descriptions (800)", "#157c80"),
        ("fsd_pruned", "FSD-pruned words (2,000)", "#cd7443"),
        ("common_words", "Frequent words (800)", "#906bb1"),
        ("random_vectors", "Gaussian vectors (800)", "#718092"),
    ]:
        group = sorted(
            [r for r in rows if r["series"] == key], key=lambda r: int(r["k"])
        )
        ax.plot(
            [int(r["k"]) for r in group],
            [100 * float(r["accuracy"]) for r in group],
            label=label,
            color=color,
            lw=1.5,
        )
    ax.axhline(
        100 * float(rows[0]["uncompressed_accuracy"]),
        color="#303846",
        ls="--",
        lw=1,
        label="Uncompressed head contributions",
    )
    ax.set(
        xlabel="Textual directions per head, K",
        ylabel="Top-1 accuracy (%)",
        xticks=range(0, 33, 4),
        xlim=(0, 32),
        ylim=(0, 80),
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.15)
    save(fig, "reconstruction")
    edits = sorted(
        table(source / "edits.csv"),
        key=lambda r: float(r["target_drop_pp"]),
        reverse=True,
    )
    random = table(source / "random-edits.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    y = np.arange(len(edits))
    for offset, key, label, color, marker in [
        (-0.16, "target_drop_pp", "Target class, our edit", "#157c80", "o"),
        (
            0,
            "random_mean_target_drop_pp",
            "Target class, random edits mean",
            "#cf753e",
            "D",
        ),
        (0.16, "other_pool_drop_pp", "Other classes, our edit", "#8563a9", "s"),
    ]:
        ax.scatter(
            [float(r[key]) for r in edits],
            y + offset,
            color=color,
            marker=marker,
            s=23,
            label=label,
            zorder=3,
        )
    for j, row in enumerate(edits):
        values = [
            float(r["target_drop_pp"]) for r in random if r["label"] == row["label"]
        ]
        lo, hi = np.percentile(values, [5, 95])
        ax.plot([lo, hi], [j, j], color="#cf753e", lw=1.3, zorder=2)
        ax.plot([lo, lo], [j - 0.055, j + 0.055], color="#cf753e", lw=1)
        ax.plot([hi, hi], [j - 0.055, j + 0.055], color="#cf753e", lw=1)
    ax.axvline(0, color="#a5acb6", lw=0.8)
    ax.set_yticks(
        y, [r["label"].replace("_", " ") + f" ({r['heads']} heads)" for r in edits]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy drop (percentage points); negative = improvement")
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=1,
    )
    ax.grid(axis="x", alpha=0.15)
    save(fig, "class-group-ablation")
    # The explanatory table accompanies both plot-only and full reproductions.
    (output / "README.md").write_text(
        "Cumulative removal: jointly replace the direct MHA and MLP contributions by fit means; the random-order line is the exact average over every same-size subset. Shading is the 5th–95th percentile over subsets.\n\nReconstruction: each head retains its mean plus its signed projection on K fitted directions; all non-head terms are replaced by their Clotho means. This is an offline output edit, not an internal intervention.\n\nClass-group ablation: internally replace selected attended-value slices by position-specific Clotho means and rerun downstream computation. Random controls use the frozen same-size complement groups; their 5th–95th percentiles are descriptive spread over ten draws, not a confidence interval. Other classes pool all non-target clips. Positive means worse accuracy.\n"
    )


def atlas(output):
    """Export all ranks, with mappings confined to the frozen top-three descriptors."""
    output = Path(output)
    mapping = {
        r["phrase_id"]: r for r in read_json(ROOT / "assets/mapping.json")["mappings"]
    }
    selected = read_json(ROOT / "assets/selected-heads.json")["heads"]
    summaries = {(r["block"], r["head"]): r["summary"] for r in selected}
    rows = table(output / "results/explanations.csv")
    heads = []
    for h in range(64):
        entries = []
        for row in rows:
            if int(row["block"]) * 32 + int(row["head"]) != h:
                continue
            rank = int(row["rank"])
            m = mapping.get(row["phrase_id"]) if rank <= 3 else None
            entries.append(
                dict(
                    rank=rank,
                    id=row["phrase_id"],
                    text=row["text"],
                    explained_variance=float(row["explained_variance"]),
                    labels=m["urbansound_labels"] if m is not None else None,
                    rationale=m.get("rationale", "") if m else "",
                )
            )
        heads.append(
            dict(
                block=h // 32,
                head=h % 32,
                summary=summaries.get((h // 32, h % 32)),
                entries=entries,
            )
        )
    save_json(
        output / "atlas.json",
        dict(
            heads=heads,
            audio_selection="Largest positive centered coefficient on the fitted direction among CC0 Clotho clips; distinct sources within each head. Illustrative, not literal caption validation.",
        ),
    )


def audio_examples(output, data, site):
    """Publish one CC0 example per top-three direction, with source/license records.

    Positive signed coefficients are selected on Clotho fit audio, not test audio.
    A global audio file is reused when several heads select the same recording.
    """
    import subprocess
    from .data import cohort, load_waveform, sha256
    from .experiment import shards

    output, site = Path(output), Path(site)
    site.mkdir(parents=True, exist_ok=True)
    atlas_data = read_json(output / "atlas.json")
    fit_rows = cohort()[0]
    basis = np.stack(
        [np.load(output / "spans/llm" / f"{h:02d}.npz")["basis"] for h in range(64)]
    )
    with np.load(output / "means.npz") as z:
        means = z["heads"].copy()
    coefficients = []
    energy = np.zeros(64, dtype=np.float64)
    for z in shards(output, "fit"):
        centered = z["heads"].astype(np.float64) - means
        coefficients.append(np.einsum("nhd,hdk->nhk", centered, basis))
        energy += np.sum(centered**2, axis=(0, 2))
    coefficients = np.concatenate(coefficients)
    variance = np.cumsum(np.sum(coefficients**2, axis=0), axis=1) / energy[:, None]
    eligible = np.array(["/publicdomain/zero/" in row["license"] for row in fit_rows])
    if not eligible.any():
        raise ValueError("No CC0 audio available for redistribution")
    audio_dir = site / "audio"
    audio_dir.mkdir(exist_ok=True)
    credits = {}
    for h, head in enumerate(atlas_data["heads"]):
        sources = set()
        for k, entry in enumerate(head["entries"]):
            entry["explained_variance"] = float(variance[h, k])
            if k >= 3:
                continue
            order = np.argsort(-coefficients[:, h, k], kind="stable")
            j = next(
                int(j)
                for j in order
                if eligible[j] and fit_rows[j]["source_id"] not in sources
            )
            row = fit_rows[j]
            sources.add(row["source_id"])
            if coefficients[j, h, k] <= 0:
                raise ValueError("No positive CC0 example for this direction")
            name = row["sample_id"] + ".mp3"
            destination = audio_dir / name
            if name not in credits:
                path = Path(data) / row["path"]
                if sha256(path) != row["sha256"]:
                    raise ValueError("Audio changed")
                wave, _ = load_waveform(path)
                start = max(0, (len(wave) - 480000) // 2)
                wave = wave[start : start + 480000]
                # Same int16 round-trip as the model; no loudness selection or amplification.
                wave = (np.clip(wave, -1, 1) * 32767).astype(np.int16).astype(
                    np.float32
                ) / 32767
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "f32le",
                        "-ar",
                        "48000",
                        "-ac",
                        "1",
                        "-i",
                        "pipe:0",
                        "-codec:a",
                        "libmp3lame",
                        "-b:a",
                        "64k",
                        "-ar",
                        "32000",
                        str(destination),
                    ],
                    input=wave.astype("<f4").tobytes(),
                    check=True,
                )
                credits[name] = dict(
                    sample_id=row["sample_id"],
                    filename=Path(row["path"]).name,
                    source_id=row["source_id"],
                    source_url=f"https://freesound.org/s/{row['source_id']}/",
                    license=row["license"],
                    start_seconds=start / 48000,
                    duration_seconds=len(wave) / 48000,
                    original_sha256=row["sha256"],
                    preview_sha256=sha256(destination),
                )
            entry["audio"] = dict(
                file="audio/" + name,
                source_url=credits[name]["source_url"],
                filename=credits[name]["filename"],
                license=row["license"],
                coefficient=float(coefficients[j, h, k]),
                sample_id=row["sample_id"],
            )
    save_json(site / "atlas.json", atlas_data)
    save_json(
        site / "audio/credits.json",
        dict(selection=atlas_data["audio_selection"], files=credits),
    )
    print(f"Published {len(credits)} CC0 previews for 192 head/rank slots", flush=True)
