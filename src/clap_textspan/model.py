"""Bounded LAION HTSAT-tiny adapter; upstream source remains unmodified."""

from contextlib import ExitStack
import json
import hashlib
from pathlib import Path
from unittest.mock import patch

from .data import DURATION_POLICY, load_waveform
from . import ROOT
import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel


def load_model(checkpoint, *, device="cpu", text_directory=None):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"Expected CPU or CUDA device, got {device}")
    # Import the pinned factory without the convenience wrapper, whose __init__
    # eagerly downloads three unused tokenizers. Upstream files stay unmodified.
    import importlib.util
    import sys
    import types

    if "laion_clap" not in sys.modules:
        spec = importlib.util.find_spec("laion_clap")
        if spec is None:
            raise RuntimeError(
                "Pinned CLAP source missing; run `python -m clap_textspan setup`"
            )
        package = types.ModuleType("laion_clap")
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        sys.modules["laion_clap"] = package
    from laion_clap.clap_module.factory import create_model

    # The checkpoint already contains ALL RoBERTa weights. Construct the exact
    # base architecture from its pinned configuration locally instead of downloading redundant text weights.
    # Strict loading below makes missing or incompatible weights fatal.
    sources = json.loads((ROOT / "assets/protocol.json").read_text())
    config_bytes = (
        Path(text_directory or Path(checkpoint).parent / "roberta-base") / "config.json"
    ).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != sources["roberta_config_sha256"]:
        raise ValueError("RoBERTa configuration hash mismatch")
    text_config = RobertaConfig.from_dict(json.loads(config_bytes))
    with patch.object(
        RobertaModel,
        "from_pretrained",
        side_effect=lambda *a, **kw: RobertaModel(text_config),
    ):
        model, config = create_model(
            "HTSAT-tiny",
            "roberta",
            precision="fp32",
            device=torch.device("cpu"),
            enable_fusion=False,
        )
    # Legacy training metadata contains a NumPy float64 scalar. Keep the
    # weights-only unpickler and allow only these inspected metadata types.
    with torch.serialization.safe_globals(
        [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    ):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    if next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    # Upstream factory.load_state_dict performs this same compatibility removal.
    removed = [k for k in state if k == "text_branch.embeddings.position_ids"]
    for key in removed:
        del state[key]
    model.load_state_dict(state, strict=True)
    model.eval().to(device=device, dtype=torch.float32)
    architecture = inspect_architecture(model)
    architecture["checkpoint_loading"] = (
        "weights_only=True, mmap=True; inspected NumPy scalar/dtype/Float64DType allowlist"
    )
    architecture["checkpoint_compatibility_removed_buffers"] = removed
    architecture["roberta_config_provenance"] = {
        k: v for k, v in sources.items() if k.startswith("roberta")
    }
    architecture["text_initialization"] = (
        "pinned roberta-base config; every persistent weight strictly loaded from CLAP checkpoint"
    )
    return model, config, architecture


def inspect_architecture(model):
    device = next(model.parameters()).device
    a = model.audio_branch
    depths = [len(layer.blocks) for layer in a.layers]
    heads = [layer.blocks[0].attn.num_heads for layer in a.layers]
    final = a.layers[3]
    checks = {
        "nonfusion": model.enable_fusion is False,
        "htsat_tiny": depths == [2, 2, 6, 2] and heads == [4, 8, 16, 32],
        "final_no_merge": final.downsample is None,
        "final_grid": all(tuple(b.input_resolution) == (8, 8) for b in final.blocks),
        "final_window": all(
            b.window_size == 8 and b.shift_size == 0 for b in final.blocks
        ),
        "final_width": all(b.attn.qkv.in_features == 768 for b in final.blocks),
        "square_spectrogram": a.spec_size == 256,
        "readout_folding": tuple(a.patch_stride) == (4, 4) and a.freq_ratio == 4,
        "audio_preprocessing": all(
            getattr(model.audio_cfg, k) == v
            for k, v in dict(
                sample_rate=48000,
                clip_samples=480000,
                mel_bins=64,
                window_size=1024,
                hop_size=480,
                fmin=50,
                fmax=14000,
            ).items()
        ),
        "relu_projection": isinstance(model.audio_projection[1], torch.nn.ReLU),
        "projection_shapes": tuple(model.audio_projection[0].weight.shape) == (512, 768)
        and tuple(model.audio_projection[2].weight.shape) == (512, 512),
        "float32_single_device": device.type in ("cpu", "cuda")
        and all(
            p.dtype == torch.float32 and p.device == device for p in model.parameters()
        ),
        "eval": not any(m.training for m in model.modules()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Unsupported CLAP architecture; expected non-fusion HTSAT-tiny/ReLU: {checks}"
        )
    return dict(
        checks=checks,
        device=str(device),
        depths=depths,
        heads=heads,
        final_grid=[8, 8],
        final_head_width=24,
        projection=str(model.audio_projection),
        parameter_count=sum(p.numel() for p in model.parameters()),
        audio_config=vars(model.audio_cfg),
        normalization="torch.nn.functional.normalize p=2 dim=-1 eps=1e-12",
    )


def prepare_audio(
    path,
    config,
    *,
    policy=DURATION_POLICY,
    diagnostic=None,
    return_details=False,
    center_crop=False,
):
    waveform, decode_details = load_waveform(path, policy=policy)
    source_samples = len(waveform)
    crop_start = 0
    if diagnostic == "short":
        # Deterministic engineering stress input: highest-energy nonoverlapping
        # 100 ms window, so the short check does not duplicate leading silence.
        count = len(waveform) // 4800
        if count == 0:
            raise ValueError(
                "Short diagnostic requires at least 100 ms of source audio"
            )
        windows = waveform[: count * 4800].reshape(count, 4800)
        crop_start = (
            int(np.argmax(np.mean(windows.astype(np.float64) ** 2, axis=1))) * 4800
        )
        waveform = waveform[crop_start : crop_start + 4800]
    elif diagnostic == "silence":
        waveform = np.zeros(4800, dtype=np.float32)
    if center_crop and len(waveform) > 480000:
        crop_start = (len(waveform) - 480000) // 2
        waveform = waveform[crop_start : crop_start + 480000]
    if len(waveform) > 480000:
        raise ValueError(
            "T1 only accepts <=10 s inputs; record an explicit crop before using longer audio"
        )
    details = dict(
        decode_details,
        source_samples=source_samples,
        used_samples=len(waveform),
        crop_start_samples=crop_start,
        repeat_count=480000 // len(waveform),
        right_zero_pad_samples=480000 % len(waveform),
        rms_before_padding=float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
        crop_policy=(
            "highest-energy nonoverlapping 100ms"
            if diagnostic == "short"
            else "center_crop_480000_repeatpad_v1" if center_crop else "none"
        ),
    )
    # Exact upstream float32 -> int16 -> float32 and repeatpad, restricted to
    # <=10 s unfused inputs. Avoid importing the unrelated training data loader.
    waveform = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).astype(
        np.float32
    ) / 32767.0
    tensor = torch.from_numpy(waveform).float()
    if len(tensor) < 480000:
        tensor = tensor.repeat(480000 // len(tensor))
        tensor = torch.nn.functional.pad(tensor, (0, 480000 - len(tensor)))
    features = dict(waveform=tensor, longer=torch.tensor([False]))
    if (
        features["waveform"].shape != (480000,)
        or features["waveform"].dtype != torch.float32
        or bool(features["longer"].any())
    ):
        raise RuntimeError("Unexpected upstream waveform preprocessing")
    return (features, details) if return_details else features


def _audio_result(model, compute, capture=False, *, prefix=True, batch_size=1):
    tensors = {}

    def pre(name):
        def hook(module, inputs):
            tensors[name] = inputs[0].detach().clone()

        return hook

    with ExitStack() as hooks:
        if capture:
            sites = {
                "stage_entry": model.audio_branch.layers[3],
                "final_pre_norm": model.audio_branch.norm,
                "pooled_latent": model.audio_projection,
                "projection_hidden": model.audio_projection[2],
            }
            if prefix:
                sites["spectrogram_image"] = model.audio_branch.patch_embed
            sites.update(
                {
                    f"block_{i}_attended_values": b.attn.proj
                    for i, b in enumerate(model.audio_branch.layers[3].blocks)
                }
            )
            for name, module in sites.items():
                hooks.callback(module.register_forward_pre_hook(pre(name)).remove)

            def mlp_output(name):
                def hook(module, inputs, output):
                    tensors[name] = output.detach().clone()

                return hook

            for i, block in enumerate(model.audio_branch.layers[3].blocks):
                hooks.callback(
                    block.mlp.register_forward_hook(
                        mlp_output(f"block_{i}_mlp_output")
                    ).remove
                )

            def projected(module, inputs, output):
                tensors["unnormalized_embedding"] = output.detach().clone()

            hooks.callback(
                model.audio_projection.register_forward_hook(projected).remove
            )
        with torch.inference_mode():
            embedding = compute().detach()
    if capture:
        expected = {
            "spectrogram_image": (1, 1, 256, 256),
            "stage_entry": (1, 64, 768),
            "final_pre_norm": (1, 64, 768),
            "pooled_latent": (1, 768),
            "projection_hidden": (1, 512),
            "unnormalized_embedding": (1, 512),
            "block_0_attended_values": (1, 64, 768),
            "block_1_attended_values": (1, 64, 768),
            "block_0_mlp_output": (1, 64, 768),
            "block_1_mlp_output": (1, 64, 768),
        }
        if not prefix:
            del expected["spectrogram_image"]
        for name, shape in expected.items():
            if (
                tuple(tensors[name].shape) != (batch_size, *shape[1:])
                or tensors[name].dtype != torch.float32
                or not torch.isfinite(tensors[name]).all()
            ):
                raise RuntimeError(
                    f"Unsupported runtime tensor {name}: {tensors[name].shape}"
                )
    if (
        tuple(embedding.shape) != (batch_size, 512)
        or embedding.dtype != torch.float32
        or not torch.isfinite(embedding).all()
    ):
        raise RuntimeError("Invalid real embedding")
    norms = torch.linalg.vector_norm(embedding, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0):
        raise RuntimeError("Expected unit normalized audio embeddings")
    return embedding.cpu().numpy(), tensors


def forward(model, features, capture=False):
    embedding, tensors = forward_batch(model, [features], capture=capture)
    return embedding[0], tensors


def forward_batch(model, features, capture=False):
    if not features:
        raise ValueError("Audio batch is empty")
    return _audio_result(
        model,
        lambda: model.get_audio_embedding(features),
        capture,
        batch_size=len(features),
    )


REPLAY_VERSION = "htsat_tiny_final_stage_embedding_v1"


def replay_final_stage(model, stage_entry, capture=False):
    """Replay every operation on the contrastive-embedding path after stage entry.

    The auxiliary framewise/classification branch is not used by CLAP's embedding.
    Preserve upstream token folding/pooling order for float32 parity.
    """
    embedding, tensors = replay_final_stage_batch(model, stage_entry, capture=capture)
    if len(embedding) != 1:
        raise ValueError("Single-clip replay requires exactly one row")
    return embedding[0], tensors


def replay_final_stage_batch(model, stage_entry, capture=False):
    inspect_architecture(model)
    device = next(model.parameters()).device
    if isinstance(stage_entry, np.ndarray):
        if stage_entry.dtype != np.float32:
            raise ValueError("Stage entry must be float32")
        stage_entry = torch.from_numpy(stage_entry)
    if (
        not isinstance(stage_entry, torch.Tensor)
        or stage_entry.ndim != 3
        or not len(stage_entry)
        or stage_entry.shape[1:] != (64, 768)
        or stage_entry.dtype != torch.float32
        or not torch.isfinite(stage_entry).all()
    ):
        raise ValueError("Replay requires finite float32 [batch,64,768] tokens")
    stage_entry = stage_entry.to(device)

    def compute():
        x, _ = model.audio_branch.layers[3](stage_entry.clone())
        return readout_final_tokens(model, x)

    return _audio_result(
        model, compute, capture, prefix=False, batch_size=len(stage_entry)
    )


def readout_final_tokens(model, x):
    """Differentiable contrastive readout from the final pre-normalization tokens."""
    a = model.audio_branch
    x = a.norm(x)
    batch, _, channels = x.shape
    sf = a.spec_size // (2 ** (len(a.depths) - 1)) // a.patch_stride[0]
    st = a.spec_size // (2 ** (len(a.depths) - 1)) // a.patch_stride[1]
    x = x.permute(0, 2, 1).contiguous().reshape(batch, channels, sf, st)
    frequency_bins = sf // a.freq_ratio
    x = x.reshape(batch, channels, sf // frequency_bins, frequency_bins, st)
    x = (
        x.permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, channels, frequency_bins, -1)
    )
    pooled = torch.flatten(a.avgpool(torch.flatten(x, 2)), 1)
    projected = model.audio_projection(pooled)
    return torch.nn.functional.normalize(projected, p=2, dim=-1, eps=1e-12)
