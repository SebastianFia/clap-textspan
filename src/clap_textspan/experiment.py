"""One explicit pipeline: audio extraction, fitting, reconstruction and internal edits."""

from contextlib import nullcontext
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from . import ROOT
from .data import cohort, read_json, save_json, sha256, write_csv


def load_runtime(checkpoint, device):
    import torch
    from .model import load_model

    torch.set_num_threads(4)
    torch.manual_seed(4518)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return load_model(checkpoint, device=device)


def feature_batch(records, data, audio, fit=False):
    from .model import prepare_audio

    return [
        prepare_audio(Path(data) / row["path"], audio, center_crop=fit)
        for row in records
    ]


def encode(model, directory, strings, batch=64):
    import torch
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(str(directory), local_files_only=True)
    result = []
    for start in range(0, len(strings), batch):
        tokens = tokenizer(
            strings[start : start + batch],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if tokens["input_ids"].shape[1] > 77:
            raise ValueError("Text would truncate")
        with torch.inference_mode():
            vectors = model.get_text_embedding(
                {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}
            )
            result.append(torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy())
    return np.concatenate(result)


def smoke(checkpoint, data, output, device="cuda"):
    """Two real audios, single/batch, stage replay, independent sums and no-op edit."""
    import torch
    from .accounting import capture, account, account_heads, replace_all
    from .model import replay_final_stage_batch

    model, audio, architecture = load_runtime(checkpoint, device)
    fit, rows, keep, targets = cohort()
    features = feature_batch(rows[:2], data, audio)
    clean, obs = capture(model, features=features)
    terms, names, errors, diagnostics = account(model, clean, obs)
    heads = account_heads(model, clean, obs)
    replay, _ = replay_final_stage_batch(model, obs["stage_entry"])
    np.testing.assert_allclose(clean, replay, atol=1e-6, rtol=1e-5)
    with replace_all(model, np.zeros((64, 64, 24), np.float32), selected=[]):
        identity, _ = replay_final_stage_batch(model, obs["stage_entry"])
    np.testing.assert_array_equal(identity, replay)
    with replace_all(
        model, np.zeros((64, 64, 24), np.float32), selected=[0, 35]
    ) as calls:
        _, changed = replay_final_stage_batch(model, obs["stage_entry"], capture=True)
    if calls != [1, 1]:
        raise ValueError("Intervention sites did not each run once")
    for b, h in [(0, 0), (1, 3)]:
        actual = changed[f"block_{b}_attended_values"].reshape(-1, 64, 32, 24)[:, :, h]
        if not torch.equal(actual, torch.zeros_like(actual)):
            raise ValueError("Selected head was not replaced")
    np.testing.assert_array_equal(
        changed["block_0_attended_values"]
        .reshape(-1, 64, 32, 24)[:, :, 1:]
        .cpu()
        .numpy(),
        obs["block_0_attended_values"].reshape(-1, 64, 32, 24)[:, :, 1:].cpu().numpy(),
    )
    single, other = capture(model, features=features[:1])
    one = account_heads(model, single, other)
    np.testing.assert_allclose(one["heads"], heads["heads"][:1], atol=1e-6, rtol=0)
    # CPU check replays the same observed stage-0 tokens; no audio preprocessor variation.
    model.cpu()
    cpu, cpu_obs = capture(model, entry=obs["stage0_entry"].cpu())
    cpu_terms, _, _, _ = account(model, cpu, cpu_obs)
    np.testing.assert_allclose(cpu, clean, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(cpu_terms, terms, atol=1e-6, rtol=0)
    model.to(device)
    classes = read_json(ROOT / "assets/protocol.json")["vocabulary"]
    text = encode(
        model, Path(checkpoint).parent / "roberta-base", [x["prompt"] for x in classes]
    )
    with np.load(ROOT / "assets/text-embeddings.npz") as z:
        np.testing.assert_allclose(text, z["classes"], atol=1e-6, rtol=1e-5)
    record = dict(
        architecture=architecture,
        accounting=errors,
        diagnostics=diagnostics,
        replay_max_error=float(np.max(np.abs(clean - replay))),
        cpu_vs_requested_device_max_error=float(np.max(np.abs(cpu - clean))),
        cpu_vs_requested_device_component_max_error=float(
            np.max(np.abs(cpu_terms - terms))
        ),
        comparison_device=str(device),
        no_op_bit_exact=True,
        sample_ids=[r["sample_id"] for r in rows[:2]],
        tolerances=dict(
            embedding_atol=1e-6, embedding_rtol=1e-5, contribution_atol=1e-6
        ),
    )
    save_json(Path(output) / "SMOKE.json", record)
    print(json.dumps(record, indent=2), flush=True)


def run_binding(checkpoint, data, batch, device):
    import torch

    # Data absolute location is not semantic; waveform hashes, order and code are.
    files = list((ROOT / "assets").glob("*")) + list(
        (ROOT / "src/clap_textspan").glob("*.py")
    )
    return dict(
        files={str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
        checkpoint=sha256(checkpoint),
        batch_size=batch,
        method="observed_forward_affine_v1",
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            requested_device=str(device),
            gpu=(
                torch.cuda.get_device_name(device)
                if str(device).startswith("cuda")
                else None
            ),
            platform=platform.platform(),
            inference_dtype="float32",
            threads=4,
        ),
    )


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def extract(checkpoint, data, output, device="cuda", batch=8):
    """Raw-waveform extraction, resumable hash-bound shards. Only eval stage entries are cached."""
    from .accounting import capture, account, account_heads

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / "assets/protocol.json")
    binding = run_binding(checkpoint, data, batch, device)
    if binding["checkpoint"] != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    receipt = output / "RUN.json"
    if receipt.exists() and read_json(receipt) != binding:
        raise ValueError("Code/inputs/environment changed; use a new output directory")
    save_json(receipt, binding)
    if not (output / "SMOKE.json").exists():
        smoke(checkpoint, data, output, device)
    model, audio, _ = load_runtime(checkpoint, device)
    fit_rows, eval_rows, _, _ = cohort()
    with np.load(ROOT / "assets/text-embeddings.npz") as z:
        texts = z["classes"].astype(np.float64)
    # Frozen embeddings are convenience assets, not opaque prerequisites: re-encode
    # every textual bank and verify it before the expensive audio pass.
    dictionary = [
        json.loads(l)["text"]
        for l in (ROOT / "assets/dictionary.jsonl").read_text().splitlines()
    ]
    fsd = [
        json.loads(l)["text"]
        for l in (ROOT / "assets/fsd-pruned.jsonl").read_text().splitlines()
    ]
    strings = {
        "llm": dictionary,
        "words": (ROOT / "assets/common-words.txt").read_text().splitlines(),
        "fsd": fsd,
    }
    with np.load(ROOT / "assets/text-embeddings.npz") as bank:
        for name, phrases in strings.items():
            actual = encode(model, Path(checkpoint).parent / "roberta-base", phrases)
            np.testing.assert_allclose(actual, bank[name], atol=1e-6, rtol=1e-5)
    for partition, records in [("fit", fit_rows), ("eval", eval_rows)]:
        for start in range(0, len(records), batch):
            rows = records[start : start + batch]
            path = output / partition / f"{start:06d}.npz"
            note = path.with_suffix(".json")
            ids = [r["sample_id"] for r in rows]
            if path.exists() and note.exists():
                saved = read_json(note)
                if saved["sha256"] != sha256(path) or saved["ids"] != ids:
                    raise ValueError(f"Changed shard {path}")
                continue
            for row in rows:
                if sha256(Path(data) / row["path"]) != row["sha256"]:
                    raise ValueError(f"Changed audio: {row['sample_id']}")
            clean, obs = capture(
                model, features=feature_batch(rows, data, audio, fit=partition == "fit")
            )
            terms, names, errors, diagnostics = account(model, clean, obs)
            result = account_heads(model, clean, obs)
            arrays = dict(
                ids=ids,
                clean=clean,
                heads=result["heads"],
                nonheads=result["nonheads"],
                term_scores=terms.astype(np.float64) @ texts.T,
                term_names=names,
            )
            if partition == "fit":
                arrays["term_sum"] = terms.sum(0, dtype=np.float64)
                values = np.stack(
                    [
                        obs[f"block_{b}_attended_values"]
                        .cpu()
                        .numpy()
                        .reshape(-1, 64, 32, 24)
                        for b in range(2)
                    ],
                    axis=1,
                )
                arrays["position_sum"] = (
                    values.sum(0, dtype=np.float64)
                    .transpose(0, 2, 1, 3)
                    .reshape(64, 64, 24)
                )
            else:
                arrays["entry"] = obs["stage_entry"].cpu().numpy()
            atomic_npz(path, **arrays)
            save_json(
                note,
                dict(
                    sha256=sha256(path),
                    ids=ids,
                    accounting=errors,
                    head_accounting=result["errors"].tolist(),
                    diagnostics=diagnostics,
                ),
            )
            if start % (batch * 16) == 0:
                print(f"{partition}: {start+len(rows)}/{len(records)}", flush=True)
    save_json(
        output / "EXTRACTED.json",
        dict(run=sha256(receipt), fit=len(fit_rows), evaluation=len(eval_rows)),
    )


def shards(output, partition):
    records = cohort()[0 if partition == "fit" else 1]
    position = 0
    for path in sorted((Path(output) / partition).glob("*.npz")):
        if sha256(path) != read_json(path.with_suffix(".json"))["sha256"]:
            raise ValueError(f"Changed shard {path}")
        with np.load(path, allow_pickle=False) as z:
            ids = z["ids"].tolist()
            if ids != [r["sample_id"] for r in records[position : position + len(ids)]]:
                raise ValueError("Shard order/IDs changed")
            yield z
            position += len(ids)
    if position != len(records):
        raise ValueError(
            f"Incomplete {partition} extraction: {position}/{len(records)}"
        )


def fit_spans(output):
    from .span import support, fit

    output = Path(output)
    # One dense 0.78 GB fit target, independent of the evaluation clips.
    values = np.concatenate([z["heads"] for z in shards(output, "fit")])
    nonheads = np.concatenate([z["nonheads"] for z in shards(output, "fit")])
    total = sum(z["term_sum"] for z in shards(output, "fit")) / len(values)
    positional = (
        sum(z["position_sum"] for z in shards(output, "fit")) / len(values)
    ).astype(np.float32)
    atomic_npz(
        output / "means.npz",
        heads=values.mean(0, dtype=np.float64),
        nonheads=nonheads.mean(0, dtype=np.float64),
        terms=total,
        positions=positional,
    )
    banks = np.load(ROOT / "assets/text-embeddings.npz")
    for h in range(64):
        target = values[:, h].astype(np.float64)
        subspace = support(target)
        for name in ["llm", "words", "gaussian", "fsd"]:
            path = output / "spans" / name / f"{h:02d}.npz"
            result = fit(target, banks[name], subspace=subspace)
            atomic_npz(path, **result)
        print(f"Fit head {h//32}/{h%32:02d} (all four banks)", flush=True)
    save_json(
        output / "FITTED.json",
        dict(
            extraction=sha256(output / "EXTRACTED.json"),
            means=sha256(output / "means.npz"),
        ),
    )


def reconstruction(output):
    from .span import projection_scores

    output = Path(output)
    _, _, keep, targets = cohort()
    with np.load(ROOT / "assets/text-embeddings.npz") as z:
        texts = z["classes"].astype(np.float64)
    with np.load(output / "means.npz") as z:
        means = z["heads"].copy()
        remainder = z["nonheads"].sum(0)
    heads = np.concatenate([z["heads"] for z in shards(output, "eval")]).astype(
        np.float64
    )
    clean = (
        np.concatenate([z["clean"] for z in shards(output, "eval")]).astype(np.float64)
        @ texts.T
    )
    base = (remainder + means.sum(0)) @ texts.T
    uncompressed = (heads.sum(1) + remainder) @ texts.T
    records = []
    atlas = []
    dictionary = [
        json.loads(l)
        for l in (ROOT / "assets/dictionary.jsonl").read_text().splitlines()
    ]
    for name, size in [("llm", 800), ("words", 800), ("gaussian", 800), ("fsd", 2000)]:
        scores = np.broadcast_to(base, (33, len(heads), 10)).copy()
        for h in range(64):
            with np.load(output / "spans" / name / f"{h:02d}.npz") as z:
                basis = z["basis"]
                indices = z["indices"]
                ev = z["explained"]
                scores += projection_scores(heads[:, h] - means[h], basis, texts)
                if name == "llm":
                    for k, index in enumerate(indices):
                        row = dictionary[int(index)]
                        atlas.append(
                            dict(
                                block=h // 32,
                                head=h % 32,
                                rank=k + 1,
                                phrase_id=row["id"],
                                text=row["text"],
                                explained_variance=float(ev[k]),
                            )
                        )
        for k in range(33):
            records.append(
                dict(
                    series=(
                        "audio_v2"
                        if name == "llm"
                        else {
                            "words": "common_words",
                            "gaussian": "random_vectors",
                            "fsd": "fsd_pruned",
                        }[name]
                    ),
                    candidates=size,
                    k=k,
                    accuracy=float(
                        (scores[k].argmax(-1)[keep] == targets[keep]).mean()
                    ),
                    clips=int(keep.sum()),
                    clean_accuracy=float(
                        (clean.argmax(-1)[keep] == targets[keep]).mean()
                    ),
                    uncompressed_accuracy=float(
                        (uncompressed.argmax(-1)[keep] == targets[keep]).mean()
                    ),
                )
            )
        atomic_npz(
            output / "scores" / f"reconstruction-{name}.npz",
            scores=scores,
            targets=targets,
            keep=keep,
        )
    write_csv(output / "results/reconstruction.csv", records)
    write_csv(output / "results/explanations.csv", atlas)


def component_ablation(output):
    """Exact 4096-subset joint removal; no Monte Carlo random-order estimate."""
    from itertools import combinations
    from .accounting import BLOCKS

    output = Path(output)
    _, _, keep, targets = cohort()
    parts = [
        (z["term_scores"].copy(), z["clean"].copy(), z["term_names"].tolist())
        for z in shards(output, "eval")
    ]
    values = np.concatenate([p[0] for p in parts])
    clean_vectors = np.concatenate([p[1] for p in parts])
    names = parts[0][2]
    with np.load(ROOT / "assets/text-embeddings.npz") as z:
        texts = z["classes"].astype(np.float64)
    with np.load(output / "means.npz") as z:
        means = z["terms"] @ texts.T
    clean = clean_vectors.astype(np.float64) @ texts.T

    def indices(kind):
        return (
            [j for j, n in enumerate(names) if n.endswith("_mlp")]
            if kind == "mlp"
            else [
                j
                for j, n in enumerate(names)
                if n.endswith(("_heads", "_attention_bias"))
            ]
        )

    mlps, attention = indices("mlp"), indices("mha")
    records = []
    for cohort_name, subset in [
        ("source-disjoint", keep),
        ("full", np.ones(len(keep), bool)),
    ]:
        for condition, selected in [
            ("clean", []),
            ("all_mlps_mean", mlps),
            ("all_mha_mean", attention),
            ("all_mha_and_mlps_mean", mlps + attention),
            ("mean_only", list(range(42))),
        ]:
            score = (
                clean - values[:, selected].sum(1) + means[selected].sum(0)
                if selected
                else clean
            )
            correct = int((score.argmax(-1)[subset] == targets[subset]).sum())
            records.append(
                dict(
                    cohort=cohort_name,
                    condition=condition,
                    clips=int(subset.sum()),
                    correct=correct,
                    accuracy=correct / int(subset.sum()),
                )
            )
    write_csv(output / "results/components.csv", records)
    delta = []
    for stage, block in BLOCKS:
        index = [
            names.index(f"s{stage}b{block}_{kind}")
            for kind in ["heads", "attention_bias", "mlp"]
        ]
        delta.append(values[:, index].sum(1) - means[index].sum(0))
    delta = np.stack(delta, axis=1)
    counts = np.zeros((4096, 2), dtype=int)
    # Gray-code updates plus regular direct-sum checks bound accumulated roundoff.
    current = clean.copy()
    previous = 0
    for step in range(4096):
        mask = step ^ (step >> 1)
        if step:
            changed = mask ^ previous
            bit = changed.bit_length() - 1
            current += (-1 if mask & changed else 1) * delta[:, bit]
        if step % 32 == 0:
            selected = [j for j in range(12) if mask & (1 << j)]
            direct = clean - delta[:, selected].sum(1) if selected else clean
            np.testing.assert_allclose(current, direct, atol=1e-12, rtol=0)
            current = direct.copy()
        correct = current.argmax(-1) == targets
        counts[mask] = [correct[keep].sum(), correct.sum()]
        previous = mask
    records = []
    sizes = np.array([m.bit_count() for m in range(4096)])
    for i, (cohort_name, n) in enumerate(
        [("source-disjoint", keep.sum()), ("full", len(keep))]
    ):
        accuracy = counts[:, i] / n
        for k in range(13):
            samples = accuracy[sizes == k]
            forward = (1 << k) - 1
            reverse = 4095 ^ ((1 << (12 - k)) - 1)
            records.append(
                dict(
                    cohort=cohort_name,
                    k=k,
                    forward_mask=forward,
                    reverse_mask=reverse,
                    first_to_last=float(accuracy[forward]),
                    last_to_first=float(accuracy[reverse]),
                    random_mean=float(samples.mean()),
                    random_p05=float(np.percentile(samples, 5)),
                    random_p95=float(np.percentile(samples, 95)),
                    subsets=len(samples),
                    clips=int(n),
                )
            )
    atomic_npz(output / "scores/subsets.npz", correct=counts)
    write_csv(output / "results/removal.csv", records)


def internal_edits(checkpoint, output, device="cuda", batch=64):
    from .accounting import replace_all
    from .model import replay_final_stage_batch

    output = Path(output)
    _, _, keep, targets = cohort()
    with np.load(ROOT / "assets/text-embeddings.npz") as z:
        texts = z["classes"].astype(np.float64)
    with np.load(output / "means.npz") as z:
        positions = z["positions"].copy()
    entries = np.concatenate([z["entry"] for z in shards(output, "eval")])
    clean = (
        np.concatenate([z["clean"] for z in shards(output, "eval")]).astype(np.float64)
        @ texts.T
    )
    groups = read_json(ROOT / "assets/groups.json")["groups"]
    # Verify that frozen groups really follow this run's top-three explanations.
    mapping = {
        r["phrase_id"]: r["urbansound_labels"]
        for r in read_json(ROOT / "assets/mapping.json")["mappings"]
    }
    dictionary = [
        json.loads(l)
        for l in (ROOT / "assets/dictionary.jsonl").read_text().splitlines()
    ]
    predicted = {g["label"]: [] for g in groups}
    for h in range(64):
        with np.load(output / "spans/llm" / f"{h:02d}.npz") as z:
            ids = z["indices"][:3]
        labels = set()
        for j in ids:
            key = dictionary[int(j)]["id"]
            if key not in mapping:
                raise ValueError(
                    "New top-three caption has no frozen mapping; do not silently substitute None"
                )
            labels.update(mapping[key])
        for label in labels:
            predicted[label].append(h)
    for g in groups:
        if predicted[g["label"]] != g["heads"]:
            raise ValueError(
                "Refitted explanations changed frozen class groups; inspect before evaluation"
            )
    model, _, _ = load_runtime(checkpoint, device)
    conditions = {
        tuple(sorted(group))
        for g in groups
        for group in [g["heads"]] + g["random_heads"]
    }
    scores = {(): clean}
    for j, selected in enumerate(sorted(conditions)):
        if not selected:
            continue
        key = hashlib.sha256(json.dumps(selected).encode()).hexdigest()[:16]
        path = output / "scores" / f"edit-{key}.npz"
        if path.exists():
            with np.load(path) as z:
                if z["heads"].tolist() != list(selected):
                    raise ValueError("Edit cache collision")
                scores[selected] = z["scores"].copy()
        else:
            result = []
            for start in range(0, len(entries), batch):
                with replace_all(model, positions, selected):
                    changed, _ = replay_final_stage_batch(
                        model, entries[start : start + batch]
                    )
                result.append(changed.astype(np.float64) @ texts.T)
            scores[selected] = np.concatenate(result)
            atomic_npz(path, heads=selected, scores=scores[selected])
        print(
            f"Internal edit {j+1}/{len(conditions)}: {len(selected)} heads", flush=True
        )
    clean_correct = clean.argmax(-1) == targets
    records = []
    random_rows = []
    for g in groups:
        target = (targets == g["target_index"]) & keep
        other = (targets != g["target_index"]) & keep
        correct = scores[tuple(sorted(g["heads"]))].argmax(-1) == targets
        drops = []
        for draw, selection in enumerate(g["random_heads"]):
            random_correct = scores[tuple(sorted(selection))].argmax(-1) == targets
            drop = float(
                100 * (clean_correct[target].mean() - random_correct[target].mean())
            )
            drops.append(drop)
            random_rows.append(
                dict(
                    label=g["label"],
                    draw=draw,
                    heads=";".join(map(str, selection)),
                    target_drop_pp=drop,
                )
            )
        records.append(
            dict(
                edit="internal-mean",
                k=3,
                target_index=g["target_index"],
                label=g["label"],
                heads=len(g["heads"]),
                target_drop_pp=float(
                    100 * (clean_correct[target].mean() - correct[target].mean())
                ),
                random_mean_target_drop_pp=float(np.mean(drops)),
                random_draws=len(drops),
                other_clips=int(other.sum()),
                other_clean_accuracy=float(clean_correct[other].mean()),
                other_edited_accuracy=float(correct[other].mean()),
                other_pool_drop_pp=float(
                    100 * (clean_correct[other].mean() - correct[other].mean())
                ),
            )
        )
    records.sort(key=lambda r: r["target_drop_pp"], reverse=True)
    write_csv(output / "results/edits.csv", records)
    write_csv(output / "results/random-edits.csv", random_rows)


def compare(output):
    """Compare every numeric CSV cell to frozen references; never silently accept drift."""
    report = {}
    for name in [
        "components",
        "removal",
        "reconstruction",
        "edits",
        "random-edits",
        "descriptions",
    ]:

        def read(path):
            with Path(path).open() as f:
                return list(csv.DictReader(f))

        expected = read(ROOT / f"reference/{name}.csv")
        actual = read(
            Path(output)
            / f'results/{"explanations" if name=="descriptions" else name}.csv'
        )
        keys = {
            "components": ["cohort", "condition"],
            "removal": ["cohort", "k"],
            "reconstruction": ["series", "k"],
            "edits": ["label"],
            "random-edits": ["label", "draw"],
            "descriptions": ["block", "head", "rank"],
        }[name]

        def indexed(rows):
            return {tuple(r[k] for k in keys): r for r in rows}

        expected, actual = indexed(expected), indexed(actual)
        if set(expected) != set(actual):
            raise ValueError(f"{name} rows differ")
        max_error = 0.0
        for key, row in expected.items():
            for column, value in row.items():
                if column not in actual[key]:
                    raise ValueError(f"Missing {name}.{column}")
                try:
                    error = abs(float(value) - float(actual[key][column]))
                except ValueError:
                    if value != actual[key][column]:
                        raise ValueError(f"{name}/{key}/{column} differs")
                    continue
                max_error = max(max_error, error)
                if error > 1e-10:
                    raise ValueError(
                        f"{name}/{key}/{column}: expected {value}, got {actual[key][column]}"
                    )
        report[name] = dict(rows=len(actual), maximum_numeric_error=max_error)
    save_json(Path(output) / "VERIFIED.json", report)
    print(json.dumps(report, indent=2), flush=True)


def normalized_gaussian(output):
    """Supplement only: normalize the SAME 800 Gaussian draws, refit and rescore."""
    from .span import fit, support, projection_scores

    output = Path(output)
    _, _, keep, targets = cohort()
    with np.load(ROOT / "assets/text-embeddings.npz") as z:
        candidates = z["gaussian"].copy()
        texts = z["classes"].astype(np.float64)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    fit_values = np.concatenate([z["heads"] for z in shards(output, "fit")])
    evaluation = np.concatenate([z["heads"] for z in shards(output, "eval")])
    with np.load(output / "means.npz") as z:
        means = z["heads"].copy()
        base = (means.sum(0) + z["nonheads"].sum(0)) @ texts.T
    scores = np.broadcast_to(base, (33, len(targets), 10)).copy()
    for h in range(64):
        fitted = fit(fit_values[:, h], candidates)
        scores += projection_scores(
            evaluation[:, h].astype(np.float64) - means[h], fitted["basis"], texts
        )
        atomic_npz(output / "spans/gaussian-unit" / f"{h:02d}.npz", **fitted)
    records = [
        dict(
            k=k,
            accuracy=float((scores[k].argmax(-1)[keep] == targets[keep]).mean()),
            clips=int(keep.sum()),
            seed=4503,
            candidates=800,
        )
        for k in range(33)
    ]
    write_csv(output / "results/normalized-gaussian.csv", records)
    atomic_npz(
        output / "scores/normalized-gaussian.npz", scores=scores, candidates=candidates
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "results/reconstruction.csv").open() as f:
        old = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, label, color in [
        ("random_vectors", "Saved Gaussian (unnormalized)", "#718092"),
        ("audio_v2", "LLM descriptions", "#157c80"),
    ]:
        rows = [r for r in old if r["series"] == name]
        ax.plot(
            [int(r["k"]) for r in rows],
            [100 * float(r["accuracy"]) for r in rows],
            label=label,
            color=color,
        )
    ax.plot(
        range(33),
        [100 * r["accuracy"] for r in records],
        label="Gaussian, row-normalized",
        color="#905da8",
    )
    ax.set(xlabel="Directions per head, K", ylabel="Top-1 accuracy (%)", xlim=(0, 32))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            destination / f"normalized-gaussian.{suffix}", dpi=220, bbox_inches="tight"
        )
    plt.close(fig)
    print(
        "Normalized Gaussian supplement complete; K=3:",
        records[3]["accuracy"],
        flush=True,
    )
