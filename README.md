# TextSpan for CLAP

**Finding Textual Explanations of CLAP Audio Encoder Attention Heads**<br>
Sebastian Fia · Gabriele Santini · Elisabetta Farella<br>
University of Trento · Fondazione Bruno Kessler

**[Companion website and listenable head atlas](https://sebastianfia.github.io/clap-textspan/)**

A small, standalone reproduction of the final paper: observed-forward accounting across HTS-AT, TextSpan explanations of its 64 final-stage attention heads, and description-selected internal head ablations. This repository does not require the authors’ development workspace or an OpenAI API key.

## Reproduce

Use Linux/WSL and Python **3.12**. The reference environment is Python 3.12.3, NumPy 1.26.4, and PyTorch 2.11.0+cu126 on an 8 GB GTX 1070 Ti. Allow about **45 GB free disk** for downloaded archives, extracted datasets, the 1.86 GB checkpoint, and run caches; 16 GB system RAM is recommended. No FSD50K audio is needed.

```bash
git clone https://github.com/SebastianFia/clap-textspan.git
cd clap-textspan
python3.12 -m venv .venv
source .venv/bin/activate
# CUDA 12.6 supports the reference Pascal GPU; choose a suitable wheel for newer GPUs.
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m clap_textspan setup
python -m clap_textspan download-data
```

`setup` verifies and downloads the pinned checkpoint, RoBERTa configuration/tokenizer, and upstream CLAP commit. It installs that source with `--no-deps`; it never upgrades the model environment. `download-data` uses the official Clotho v2.1 and UrbanSound8K archives and checks the waveform hashes against the frozen manifests. Dataset downloads can be slow. Archives can be removed after successful extraction and verification.

If you already have the datasets, arrange them as listed in `assets/fit.csv` and `assets/eval.csv`, relative to a chosen `--data-root`. Clotho files are under `data/clotho/{split}/{split}/`; UrbanSound files are under `data/urbansound8k/UrbanSound8K/audio/fold*/`. Use `verify-data --data-root /path/to/root` to check existing files. The checkpoint and its `roberta-base/` directory must share a parent directory; `--checkpoint` can point there.

**Run quick checks before a full job:**

```bash
python -m unittest discover -s tests -v
python -m clap_textspan smoke --output runs/paper
# Only after the smoke checks pass:
python -m clap_textspan run --output runs/paper
```

The smoke test uses two real clips and checks independent accounting, intact/replay parity, an exactly unchanged no-op edit, single/batch agreement, CPU/GPU agreement, and class-text encoding. The full run also re-encodes all textual banks before extracting audio contributions. It runs inference only; there is no model training.

The full command starts from **all raw waveforms**, fits on all 5,929 Clotho clips, evaluates on the 7,962 UrbanSound clips with disjoint source IDs, and produces:

| Artifact under `runs/paper/` | Paper result |
| --- | --- |
| `results/components.csv` | Clean, all-MLP, all-MHA and joint direct-contribution controls |
| `results/removal.csv`, `scores/subsets.npz` | Forward/reverse cumulative removal and every one of 4,096 block subsets |
| `results/reconstruction.csv`, `spans/` | Four dictionaries, 64 heads, K=0–32; signed TextSpan fits |
| `results/edits.csv`, `results/random-edits.csv` | Internal class-group mean ablation and exact frozen random controls |
| `results/explanations.csv` | Every selected description, its rank and cumulative explained variance |
| `figures/` | Three paper-ready figures, PDF and PNG, plus explanations |
| `SMOKE.json`, `RUN.json`, `VERIFIED.json` | Checks, hashes/environment, and reference comparison |

The six qualitative heads in the paper table and their top-five descriptions are frozen in `assets/selected-heads.json` and `reference/selected-heads.csv`. No head selection is repeated after seeing new results.

`run` uses batches of **16**, matching the reference extraction, and 64 for internal replay. On smaller GPUs, reduce `--batch-size`; tiny changes near decision ties can affect subset averages. Each extraction shard records its IDs, checksum and conservation diagnostics. Resume with the **identical command, inputs, source files and environment**. An incompatible run must use a new output directory. Do not share one output directory between jobs.

The expensive audio extraction is cached. To repeat later stages independently:

```bash
python -m clap_textspan fit --output runs/paper
python -m clap_textspan evaluate --output runs/paper
python -m clap_textspan check --output runs/paper
python -m clap_textspan plots --output runs/paper
python -m clap_textspan atlas --output runs/paper
# Optional supplementary baseline; does not alter any paper reference:
python -m clap_textspan normalized-gaussian --output runs/paper
```

`check` compares **every** CSV field with the frozen reference and reports discrepancies. See [protocol and validation notes](REPRODUCIBILITY.md) for direct versus internal edits, the exact controls, tolerances, and two manuscript wording discrepancies. A failed check is not silently accepted.

## Just inspect or redraw the results

No model, dataset or GPU is required:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 matplotlib==3.10.7 setuptools==81.0.0 wheel==0.48.0
python -m pip install --no-build-isolation -e .
python -m clap_textspan plots --reference --output runs/reference
```

`reference/` contains the frozen paper CSVs, not newly simulated results. `docs/` is the buildless GitHub Pages website; serve it with `python -m http.server 8000 --directory docs`.

## Atlas and audio previews

The website contains all 64 heads and 32 ranks per head, including frozen UrbanSound mappings for ranks 1–3. Empty mappings are explicit. Six selected head summaries reproduce the qualitative paper table.

There is one listenable example for each of the first three directions. We select the largest positive **centered fitted coefficient** among CC0 Clotho recordings, without repeating a source within the same head. Previews use the same center 10-second window as model fitting, compressed as mono MP3; shared recordings are stored once. These illustrate directions, not a claim that a caption literally describes a recording. The coefficient sign is meaningful.

To regenerate the atlas and audio from your own completed run, install FFmpeg, then:

```bash
python -m clap_textspan listen --output runs/paper --data-root . --site docs
```

This writes `docs/atlas.json`, deduplicated previews, and an audio source/license manifest. It does not upload anything. GitHub Pages publishes this repository’s `main:/docs` directory, without a framework or build service. The source recordings retain their own licenses.

## Prompts and frozen inputs

`prompts/dictionary.md` is the single detailed prompt used to generate 800 descriptions. `prompts/mapping.md` is the single mapping prompt, containing the actual 155 distinct top-three descriptions and worked examples. `prompts/provenance.json` records request settings and generation receipts. Both used `gpt-5.4-mini-2026-03-17`, high reasoning, no tools. The accepted outputs are in `assets/`.

Exact reproduction uses these frozen outputs. Calling an LLM again—even the same model with the same prompt—creates a **new dictionary/mapping experiment**. Do not overwrite the frozen files or assume a new response recreates the published numbers. Prompts can also be pasted into a web chat; use the recorded JSON schema and validate the result before any new experiment.

## Layout

- `src/clap_textspan/`: one package: data/setup, checkpoint adapter, independent accounting, NumPy TextSpan, one experiment pipeline, and reporting.
- `assets/`: protocol, exact sample manifests, dictionaries, mappings, random groups, and small frozen text-vector banks (verified by re-encoding).
- `reference/`: CSV results and validation records; `prompts/`: two generation prompts.
- `docs/`: static companion website and licensed audio previews.
- `tests/`: mathematical/protocol checks. Heavy downloads and generated run caches stay outside Git.

## Acknowledgments and citation

The implementation follows [TextSpan](https://github.com/yossigandelsman/clip_text_span) (Gandelsman, Efros and Steinhardt, ICLR 2024), using the [LAION-CLAP](https://github.com/LAION-AI/CLAP) checkpoint and its HTS-AT encoder. Clotho: Drossos, Lipping and Virtanen, ICASSP 2020. UrbanSound8K: Salamon, Jacoby and Bello, ACM Multimedia 2014. FSD-pruned vocabulary: Zhang, Thomaz and Lu, IJCNN 2025, [paper](https://arxiv.org/abs/2504.14076), [source project](https://osf.io/3cgsu/).

Original release code is MIT-licensed. Dataset recordings, upstream model/code assets and external vocabulary resources retain their own terms; see [NOTICE.md](NOTICE.md). The frozen manifests identify recording sources and licenses, and the hosted previews are CC0 only.
