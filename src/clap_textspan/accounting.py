"""Independent accounting terms; this is an observed-forward readout, not a Jacobian."""

from contextlib import ExitStack, contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from .model import (
    inspect_architecture,
    forward_batch,
    _audio_result,
    readout_final_tokens,
)

DEPTHS = (2, 2, 6, 2)
BLOCKS = tuple((s, b) for s, d in enumerate(DEPTHS) for b in range(d))


def pool_tokens(terms, audio):
    # Preserve the upstream folding and reduction order, just as verified replay does.
    batch, _, channels = terms.shape
    sf = audio.spec_size // (2 ** (len(audio.depths) - 1)) // audio.patch_stride[0]
    st = audio.spec_size // (2 ** (len(audio.depths) - 1)) // audio.patch_stride[1]
    x = terms.permute(0, 2, 1).contiguous().reshape(batch, channels, sf, st)
    bins = sf // audio.freq_ratio
    x = (
        x.reshape(batch, channels, sf // bins, bins, st)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, channels, bins, -1)
    )
    return torch.flatten(audio.avgpool(torch.flatten(x, 2)), 1)


def restore_windows(writes, block, batch):
    """Undo window partition and cyclic shift for an attention residual write."""
    from laion_clap.clap_module.htsat import window_reverse

    height, width = block.input_resolution
    size = block.window_size
    if writes.shape != (
        batch * (height // size) * (width // size),
        size * size,
        block.dim,
    ):
        raise ValueError("Unexpected window attention shape")
    grid = window_reverse(
        writes.reshape(-1, size, size, block.dim), size, height, width
    )
    if block.shift_size:
        grid = torch.roll(
            grid, shifts=(block.shift_size, block.shift_size), dims=(1, 2)
        )
    return grid.reshape(batch, height * width, block.dim)


def pack_merge(x, resolution):
    height, width = resolution
    if x.ndim != 3 or x.shape[1] != height * width or height % 2 or width % 2:
        raise ValueError("Invalid patch-merge token grid")
    channels = x.shape[-1]
    grid = x.reshape(len(x), height, width, channels)
    return torch.cat(
        [
            grid[:, 0::2, 0::2],
            grid[:, 1::2, 0::2],
            grid[:, 0::2, 1::2],
            grid[:, 1::2, 1::2],
        ],
        dim=-1,
    ).reshape(len(x), height * width // 4, 4 * channels)


def merge_linear(x, observed, module, resolution):
    packed = pack_merge(x, resolution)
    scale = module.norm.weight / torch.sqrt(
        observed.var(dim=-1, unbiased=False, keepdim=True) + module.norm.eps
    )
    return F.linear(
        (packed - packed.mean(dim=-1, keepdim=True)) * scale, module.reduction.weight
    )


def check_architecture(model):
    inspect_architecture(model)
    for stage, layer in enumerate(model.audio_branch.layers):
        width = 96 * 2**stage
        grid = 64 // 2**stage
        if len(layer.blocks) != DEPTHS[stage] or any(
            b.dim != width
            or tuple(b.input_resolution) != (grid, grid)
            or b.attn.num_heads != 4 * 2**stage
            or b.window_size != 8
            for b in layer.blocks
        ):
            raise ValueError(f"Unsupported stage {stage}")
        if stage < 3:
            merge = layer.downsample
            if (
                not isinstance(merge.norm, torch.nn.LayerNorm)
                or tuple(merge.norm.normalized_shape) != (4 * width,)
                or tuple(merge.reduction.weight.shape) != (2 * width, 4 * width)
                or merge.reduction.bias is not None
            ):
                raise ValueError(f"Unsupported patch merge {stage}")


@torch.inference_mode()
def capture(model, *, features=None, entry=None):
    check_architecture(model)
    obs = {}

    def pre(name):
        def hook(module, inputs):
            obs[name] = inputs[0].detach().clone()

        return hook

    def post(name):
        def hook(module, inputs, result):
            obs[name] = result.detach().clone()

        return hook

    with ExitStack() as hooks:
        for stage, layer in enumerate(model.audio_branch.layers):
            hooks.callback(
                layer.register_forward_pre_hook(pre(f"stage{stage}_entry")).remove
            )
            if layer.downsample is not None:
                hooks.callback(
                    layer.downsample.register_forward_pre_hook(
                        pre(f"stage{stage}_final")
                    ).remove
                )
            for block, module in enumerate(layer.blocks):
                hooks.callback(
                    module.attn.proj.register_forward_pre_hook(
                        pre(f"s{stage}b{block}_values")
                    ).remove
                )
                hooks.callback(
                    module.mlp.register_forward_hook(
                        post(f"s{stage}b{block}_mlp")
                    ).remove
                )
        if entry is None:
            clean, final = forward_batch(model, features, capture=True)
        else:
            if (
                entry.dtype != torch.float32
                or tuple(entry.shape[1:]) != (4096, 96)
                or not torch.isfinite(entry).all()
            ):
                raise ValueError(
                    "Replay requires finite float32 [batch,4096,96] stage-0 entries"
                )

            def compute():
                x = entry.clone()
                for layer in model.audio_branch.layers:
                    x, _ = layer(x)
                return readout_final_tokens(model, x)

            clean, final = _audio_result(
                model, compute, capture=True, prefix=False, batch_size=len(entry)
            )
    obs.update(final)
    return clean, obs


@torch.inference_mode()
def account(model, clean, obs):
    audio = model.audio_branch
    batch = len(clean)
    errors = {}

    def check(name, actual, expected, normalized=False):
        actual, expected = actual.double(), expected.double()
        delta = (actual - expected).flatten(1)
        ref = expected.flatten(1)
        maximum = float(delta.abs().max())
        relative = float((delta.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)).max())
        bound = 1e-6 if normalized else 1e-6 + 1e-5 * float(expected.abs().max())
        if maximum > bound or relative > 1e-5:
            raise ValueError(
                f"{name} conservation failed: absolute={maximum}, relative={relative}, bound={bound}"
            )
        errors[name] = [maximum, relative]

    terms = {"initial_stage_entry": obs["stage0_entry"]}
    for stage, layer in enumerate(audio.layers):
        entry = obs[f"stage{stage}_entry"]
        check(f"stage{stage}_entry", sum(terms.values()), entry)
        for block, module in enumerate(layer.blocks):
            name = f"s{stage}b{block}"
            values = obs[name + "_values"]
            terms[name + "_heads"] = restore_windows(
                F.linear(values, module.attn.proj.weight), module, batch
            )
            terms[name + "_mlp"] = obs[name + "_mlp"]
            terms[name + "_attention_bias"] = module.attn.proj.bias.expand_as(entry)
        expected = obs[f"stage{stage}_final"] if stage < 3 else obs["final_pre_norm"]
        check(f"stage{stage}_final", sum(terms.values()), expected)
        if stage < 3:
            merge = layer.downsample
            resolution = layer.blocks[0].input_resolution
            observed = pack_merge(expected, resolution)
            terms = {
                name: merge_linear(value, observed, merge, resolution)
                for name, value in terms.items()
            }
            terms[f"merge{stage}_bias"] = F.linear(
                merge.norm.bias, merge.reduction.weight
            ).expand_as(obs[f"stage{stage+1}_entry"])
            check(f"merge{stage}", sum(terms.values()), obs[f"stage{stage+1}_entry"])
    names = list(terms)
    raw = torch.stack(list(terms.values()))
    final = obs["final_pre_norm"]
    scale = audio.norm.weight / torch.sqrt(
        final.var(dim=-1, unbiased=False, keepdim=True) + audio.norm.eps
    )
    pooled = pool_tokens(
        ((raw - raw.mean(dim=-1, keepdim=True)) * scale).flatten(0, 1), audio
    ).reshape(len(raw), batch, 768)
    first, _, second = model.audio_projection
    gates = obs["projection_hidden"] > 0
    raw_norm = obs["unnormalized_embedding"].norm(dim=-1)
    norm = raw_norm.clamp_min(1e-12)
    projected = (
        F.linear(F.linear(pooled, first.weight) * gates, second.weight)
        / norm[None, :, None]
    )
    beta = (
        F.linear(F.linear(audio.norm.bias, first.weight) * gates, second.weight)
        / norm[:, None]
    )
    offset = F.linear(first.bias * gates, second.weight, second.bias) / norm[:, None]
    projected = torch.cat([projected, beta[None], offset[None]])
    names += ["final_layernorm_bias", "projection_offset"]
    check(
        "embedding",
        projected.sum(0),
        torch.as_tensor(clean, device=projected.device),
        normalized=True,
    )
    if projected.shape != (42, batch, 512) or not torch.isfinite(projected).all():
        raise ValueError("Expected 42 independent finite accounting terms")
    diagnostics = dict(
        raw_norm_min=float(raw_norm.min()),
        component_norm_min=float(projected.norm(dim=-1).min()),
    )
    return projected.transpose(0, 1).cpu().numpy(), names, errors, diagnostics


def head_writes(model, values, block):
    # Concatenated attended values before W_O; each slice is one head.
    weight = (
        model.audio_branch.layers[3].blocks[block].attn.proj.weight.reshape(768, 32, 24)
    )
    return torch.einsum("bthd,ohd->hbto", values.reshape(-1, 64, 32, 24), weight)


@torch.inference_mode()
def account_heads(model, clean, obs):
    audio = model.audio_branch
    tokens = obs["final_pre_norm"]
    scale = audio.norm.weight / torch.sqrt(
        tokens.var(dim=-1, unbiased=False, keepdim=True) + audio.norm.eps
    )
    first, _, second = model.audio_projection
    gates = obs["projection_hidden"] > 0
    norm = torch.linalg.vector_norm(obs["unnormalized_embedding"], dim=-1).clamp_min(
        1e-12
    )

    def propagate(writes):
        # [terms,batch,tokens,channels] plus observed, clip-specific factors.
        linear = (writes - writes.mean(dim=-1, keepdim=True)) * scale
        pooled = pool_tokens(linear.flatten(0, 1), audio).reshape(
            len(writes), len(tokens), -1
        )
        return (
            pooled,
            F.linear(F.linear(pooled, first.weight) * gates, second.weight)
            / norm[None, :, None],
        )

    maxima = {}

    def check(name, expected, actual, abs_bound, rel_bound=1e-5):
        diff = (actual - expected).flatten(1)
        reference = expected.flatten(1)
        abs_error = diff.abs().amax().item()
        rel_error = (
            (
                torch.linalg.vector_norm(diff, dim=1)
                / torch.linalg.vector_norm(reference, dim=1).clamp_min(1e-12)
            )
            .amax()
            .item()
        )
        if abs_error > abs_bound or rel_error > rel_bound:
            raise ValueError(
                f"{name} accounting error: abs={abs_error}, relative={rel_error}"
            )
        maxima[name] = [abs_error, rel_error]

    raw_nonheads = [
        obs["stage_entry"],
        obs["block_0_mlp_output"],
        obs["block_1_mlp_output"],
    ]
    token_sum = sum(raw_nonheads)
    heads, pooled_sum = [], torch.zeros_like(obs["pooled_latent"])
    for b, block in enumerate(audio.layers[3].blocks):
        values = obs[f"block_{b}_attended_values"]
        writes = head_writes(model, values, b)
        bias = block.attn.proj.bias.expand_as(tokens)
        attention = writes.sum(0) + bias
        full_attention = block.attn.proj(values)
        check(
            f"attention_{b}",
            full_attention,
            attention,
            1e-6 + 1e-5 * full_attention.abs().max().item(),
        )
        token_sum = token_sum + attention
        raw_nonheads.append(bias)
        for chunk in writes.split(8):
            pooled, terms = propagate(chunk)
            pooled_sum += pooled.sum(0)
            heads.append(terms)
    check("tokens", tokens, token_sum, 1e-6 + 1e-5 * tokens.abs().max().item())
    pooled, remainder_terms = propagate(torch.stack(raw_nonheads))
    pooled_sum += pooled.sum(0) + audio.norm.bias
    expected_pool = obs["pooled_latent"]
    check(
        "pool",
        expected_pool,
        pooled_sum,
        1e-6 + 1e-5 * expected_pool.abs().max().item(),
    )
    beta = (
        F.linear(F.linear(audio.norm.bias, first.weight) * gates, second.weight)
        / norm[:, None]
    )
    offset = F.linear(first.bias * gates, second.weight, second.bias) / norm[:, None]
    remainder_terms = torch.cat(
        (remainder_terms, beta.unsqueeze(0), offset.unsqueeze(0))
    )
    head_terms = torch.cat(heads)
    reconstructed = head_terms.sum(0) + remainder_terms.sum(0)
    check(
        "embedding", torch.as_tensor(clean, device=tokens.device), reconstructed, 1e-6
    )
    if remainder_terms.shape[0] != 7:
        raise ValueError("Incomplete independent remainder")
    return dict(
        clean=clean,
        heads=head_terms.transpose(0, 1).cpu().numpy(),
        nonheads=remainder_terms.transpose(0, 1).cpu().numpy(),
        errors=np.array(list(maxima.values())),
        error_names=list(maxima),
        raw_norm_min=norm.min().item(),
    )


@contextmanager
def replace_all(model, means, selected=None):
    """Replace the selected pre-W_O head slices in both blocks in the same replay."""
    device = next(model.parameters()).device
    values = torch.as_tensor(means, device=device)
    if (
        values.shape != (64, 64, 24)
        or values.dtype != torch.float32
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Invalid positional means")
    packed = values.reshape(2, 32, 64, 24).permute(0, 2, 1, 3).reshape(2, 64, 768)
    mask = None
    if selected is not None:
        selected = tuple(selected)
        if len(set(selected)) != len(selected) or any(
            h < 0 or h >= 64 for h in selected
        ):
            raise ValueError("Invalid head selection")
        mask = torch.zeros(64, dtype=torch.bool, device=device)
        mask[list(selected)] = True
        mask = mask.reshape(2, 32).repeat_interleave(24, dim=1)
    calls = [0, 0]

    def replace(block):
        def hook(_, args):
            calls[block] += 1
            x = args[0]
            if x.shape[1:] != (64, 768) or x.dtype != torch.float32:
                raise ValueError("Unexpected attended-value layout")
            replacement = packed[block].expand_as(x)
            if mask is not None:
                replacement = torch.where(mask[block], replacement, x)
            return (replacement, *args[1:])

        return hook

    with ExitStack() as stack:
        for b, layer in enumerate(model.audio_branch.layers[3].blocks):
            stack.callback(layer.attn.proj.register_forward_pre_hook(replace(b)).remove)
        yield calls
