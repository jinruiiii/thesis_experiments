from __future__ import annotations

import ast
import json
from pathlib import Path


def parse_hidden_sizes(raw) -> list[int]:
    """Parse a Hidden Sizes cell (list/tuple string or already a sequence)."""
    if isinstance(raw, (list, tuple)):
        sizes = [int(s) for s in raw]
    else:
        sizes = ast.literal_eval(str(raw))
        if not isinstance(sizes, (list, tuple)):
            raise ValueError(f"Invalid Hidden Sizes value {raw!r}")
        sizes = [int(s) for s in sizes]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"Hidden Sizes must be positive ints, got {sizes} from {raw!r}")
    return sizes


def dense_fnn_flops(in_features: int, hidden_sizes, out_features: int) -> int:
    """
    Realized dense FLOPs for a mean-field FNN forward (batch 1).

    Counts 2 * sum_l n_{l-1} * n_l over linear weight matmuls only
    (ignore LayerNorm / activations / bias adds).
    """
    widths = [int(in_features), *[int(h) for h in hidden_sizes], int(out_features)]
    return int(2 * sum(widths[i] * widths[i + 1] for i in range(len(widths) - 1)))


def theoretical_sparse_fnn_flops(active_weight_count: int) -> int:
    """Theoretical sparse FLOPs: 2 * number of active (nonzero) weights."""
    return int(2 * int(active_weight_count))


def nest_active_weight_count_from_architecture(arch: dict) -> int | None:
    """
    Prefer sum(per_layer_active_weights); else None if unavailable.
    """
    per_layer = arch.get("per_layer_active_weights")
    if isinstance(per_layer, (list, tuple)) and len(per_layer) > 0:
        return int(sum(int(x) for x in per_layer))
    return None


def load_nest_theoretical_flops(exp_dir) -> int | None:
    """
    Read Nest final_architecture.json and return theoretical sparse FLOPs,
    or None if the file / fields are missing.
    """
    path = Path(exp_dir) / "final_architecture.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        arch = json.load(f)
    active_weights = nest_active_weight_count_from_architecture(arch)
    if active_weights is None:
        return None
    return theoretical_sparse_fnn_flops(active_weights)


def experiment_forward_flops(
    exp_dir=None,
    hidden_sizes=None,
    *,
    in_features: int = 784,
    out_features: int = 10,
    active_weight_count: int | None = None,
    prefer_nest_sparse: bool = True,
) -> int:
    """
    One forward-pass FLOPs for experiment summaries / Acc-vs-FLOPs plots.

    - If ``active_weight_count`` is given: ``2 * active_weight_count`` (Nest sparse).
    - Else if ``prefer_nest_sparse`` and ``exp_dir`` has Nest architecture JSON: use that.
    - Else: dense GEMM from ``hidden_sizes``.
    """
    if active_weight_count is not None:
        return theoretical_sparse_fnn_flops(active_weight_count)
    if prefer_nest_sparse and exp_dir is not None:
        nest_flops = load_nest_theoretical_flops(exp_dir)
        if nest_flops is not None:
            return nest_flops
    if hidden_sizes is None:
        raise ValueError(
            "hidden_sizes required when Nest sparse FLOPs are unavailable"
        )
    return dense_fnn_flops(in_features, hidden_sizes, out_features)
