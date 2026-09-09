from __future__ import annotations

import ast
import json
from pathlib import Path


def parse_conv_channels(raw) -> list[int]:
    if isinstance(raw, (list, tuple)):
        channels = [int(c) for c in raw]
    else:
        text = str(raw)
        if text.endswith("f"):
            parts = text.split("_")
            channels = [int(p[:-1]) for p in parts if p.endswith("f")]
        else:
            channels = ast.literal_eval(text)
            if not isinstance(channels, (list, tuple)):
                raise ValueError(f"Invalid Conv Channels value {raw!r}")
            channels = [int(c) for c in channels]
    if not channels or any(c <= 0 for c in channels):
        raise ValueError(f"Conv Channels must be positive ints, got {channels} from {raw!r}")
    return channels


def theoretical_sparse_cnn_flops(active_weight_count: int) -> int:
    """Theoretical sparse FLOPs: 2 * number of active (nonzero) weights."""
    return int(2 * int(active_weight_count))


def nest_active_weight_count_from_architecture(arch: dict) -> int | None:
    per_layer = arch.get("per_layer_active_weights")
    if isinstance(per_layer, (list, tuple)) and len(per_layer) > 0:
        return int(sum(int(x) for x in per_layer))
    return None


def load_nest_theoretical_flops(exp_dir) -> int | None:
    arch_path = Path(exp_dir) / "final_architecture.json"
    if not arch_path.exists():
        return None
    with open(arch_path, encoding="utf-8") as f:
        arch = json.load(f)
    active_weights = nest_active_weight_count_from_architecture(arch)
    if active_weights is None:
        return None
    return theoretical_sparse_cnn_flops(active_weights)
