"""Run `python -m clap_textspan --help` from this checkout."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the CLAP TextSpan paper. Run smoke before run."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser(
        "setup", help="Download pinned model assets and install pinned CLAP source"
    )
    setup.add_argument("--model-dir", default="checkpoints")
    download = sub.add_parser(
        "download-data", help="Download official Clotho and UrbanSound8K archives"
    )
    download.add_argument("--data-root", default=".")
    verify = sub.add_parser(
        "verify-data", help="Verify every waveform against frozen manifests"
    )
    verify.add_argument("--data-root", default=".")
    for name in ["smoke", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", default="checkpoints/630k-audioset-best.pt")
        p.add_argument("--data-root", default=".")
        p.add_argument("--output", default="runs/paper")
        p.add_argument("--device", default="cuda")
        if name == "run":
            p.add_argument("--batch-size", type=int, default=16)
    for name in [
        "fit",
        "evaluate",
        "check",
        "plots",
        "atlas",
        "listen",
        "normalized-gaussian",
    ]:
        p = sub.add_parser(name)
        p.add_argument("--output", default="runs/paper")
        if name == "listen":
            p.add_argument("--data-root", default=".")
            p.add_argument("--site", default="docs")
        if name == "plots":
            p.add_argument(
                "--reference",
                action="store_true",
                help="Use shipped CSVs; no GPU/audio/checkpoint required",
            )
    args = parser.parse_args()
    if args.command in ["setup", "verify-data", "download-data"]:
        from .data import setup, verify_data, download_data

        if args.command == "setup":
            setup(args.model_dir)
        elif args.command == "download-data":
            download_data(args.data_root)
        else:
            verify_data(args.data_root)
    elif args.command in ["smoke", "run"]:
        from .experiment import (
            smoke,
            extract,
            fit_spans,
            reconstruction,
            component_ablation,
            internal_edits,
            compare,
        )

        if args.command == "smoke":
            smoke(args.checkpoint, args.data_root, args.output, args.device)
        else:
            extract(
                args.checkpoint,
                args.data_root,
                args.output,
                args.device,
                args.batch_size,
            )
            fit_spans(args.output)
            reconstruction(args.output)
            component_ablation(args.output)
            internal_edits(args.checkpoint, args.output, args.device)
            compare(args.output)
            from .report import plots

            plots(Path(args.output) / "results", Path(args.output) / "figures")
            from .data import save_json, sha256

            root = Path(args.output)
            save_json(
                root / "COMPLETE.json",
                dict(
                    run=sha256(root / "RUN.json"),
                    verified=sha256(root / "VERIFIED.json"),
                    results={
                        p.name: sha256(p)
                        for p in sorted((root / "results").glob("*.csv"))
                    },
                ),
            )
    elif args.command == "normalized-gaussian":
        from .experiment import normalized_gaussian

        normalized_gaussian(args.output)
    elif args.command == "fit":
        from .experiment import fit_spans

        fit_spans(args.output)
    elif args.command == "evaluate":
        from .experiment import reconstruction, component_ablation

        reconstruction(args.output)
        component_ablation(args.output)
    elif args.command == "check":
        from .experiment import compare

        compare(args.output)
    elif args.command == "plots":
        from .report import plots
        from . import ROOT

        plots(
            ROOT / "reference" if args.reference else Path(args.output) / "results",
            Path(args.output) / "figures",
        )
    elif args.command == "listen":
        from .report import atlas, audio_examples

        atlas(args.output)
        audio_examples(args.output, args.data_root, args.site)
    elif args.command == "atlas":
        from .report import atlas

        atlas(args.output)


if __name__ == "__main__":
    main()
