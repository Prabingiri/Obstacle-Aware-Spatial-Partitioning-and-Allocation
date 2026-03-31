# src/baseline_huntington_hill.py   (or src/hh_baseline.py)
import math
from typing import Dict, Sequence, Any

from src.oabar.strip_perimeter import Strip
from src.oabar.optimal_axis_selection import OptimalAxisSelection


def _safe_wcrt(w: float) -> float:
    if w is None or not math.isfinite(w) or w < 0:
        return 0.0
    return float(w)


def _priority(weight: float, s: int) -> float:
    # Huntington–Hill: w / sqrt(s(s+1)), s>=1
    return weight / math.sqrt(s * (s + 1))


# ✅ MINIMAL: tuple compatibility helpers (same as greedy/RBA)
def _unpack_part(part):
    """
    Supports:
      - 4-tuple: (reg, obs, seq, valid)
      - 5-tuple: (reg, obs, seq, valid, meta)
      - 6-tuple: (reg, obs, seq, valid, meta, leaf_id)
    Returns: reg, obs, seq, valid, meta_or_none, leaf_id_or_none
    """
    if part is None or len(part) < 4:
        raise ValueError(f"Bad partition tuple (len={len(part) if part is not None else 'None'}): {part}")
    reg, obs, seq, valid = part[0], part[1], part[2], part[3]
    meta = part[4] if len(part) >= 5 and isinstance(part[4], dict) else None
    leaf_id = part[5] if len(part) >= 6 else None
    return reg, obs, seq, valid, meta, leaf_id


def _repack_part(reg, obs, seq, valid, meta, leaf_id):
    if meta is None and leaf_id is None:
        return (reg, obs, seq, valid)
    if leaf_id is None:
        return (reg, obs, seq, valid, meta)
    return (reg, obs, seq, valid, meta, leaf_id)


def _split_once(reg, obs, seq, numerical_method="brent", metric="NWCRT"):
    seq0 = [] if seq is None else list(seq)
    sel = OptimalAxisSelection(reg, obs, metric, numerical_method)
    axis, _, _, (Lr, Lo), (Rr, Ro) = sel.select_best_axis()
    left = (Lr, Lo, seq0 + [axis], True)
    right = (Rr, Ro, seq0 + [axis], True)
    return left, right


def _apply_splits_per_leaf(
    parts,
    extra_splits_per_leaf: Sequence[int],
    numerical_method="brent",
    metric="NWCRT",
):
    """
    Realize k_i extra splits inside original leaf i.

    Implementation (unchanged logic):
      subtree=[leaf], repeat k times:
        pick current worst child (max WCRT) within this subtree
        split it once -> subtree size increases by +1
      After k splits, subtree has exactly (k+1) leaves.

    ✅ Minimal fix:
      - supports OA-BAR 4/5/6-tuples
      - preserves provenance (meta + leaf_id) for all descendants
    """
    out = []
    if len(extra_splits_per_leaf) != len(parts):
        raise ValueError("extra_splits_per_leaf must have same length as parts.")

    for i, part in enumerate(parts):
        reg, obs, seq, valid, meta, leaf_id = _unpack_part(part)
        k = int(extra_splits_per_leaf[i])

        if k <= 0:
            out.append(_repack_part(reg, obs, seq, valid, meta, leaf_id))
            continue

        # subtree items keep provenance alongside geometry
        # each item: (reg, obs, seq, valid, meta, leaf_id)
        subtree = [(reg, obs, seq, valid, meta, leaf_id)]

        for _ in range(k):
            wcrts = [_safe_wcrt(Strip(r, o).calculate_region_wcrt()) for (r, o, *_rest) in subtree]
            idx = max(range(len(subtree)), key=lambda t: wcrts[t])

            r0, o0, s0, v0, m0, id0 = subtree.pop(idx)
            L, R = _split_once(r0, o0, s0, numerical_method=numerical_method, metric=metric)

            # Children inherit provenance of the original OA-BAR leaf
            Lr, Lo, Ls, Lv = L
            Rr, Ro, Rs, Rv = R
            subtree.append((Lr, Lo, Ls, Lv, m0, id0))
            subtree.append((Rr, Ro, Rs, Rv, m0, id0))

        # Emit all leaves of this expanded subtree, preserving provenance
        out.extend([_repack_part(r, o, s, v, m, lid) for (r, o, s, v, m, lid) in subtree])

    return out


def hh_apportion(
    parts,
    m_extra: int,
    beta: float = 1.0,           # unused; kept for signature consistency
    numerical_method="brent",
    metric="NWCRT",
    return_meta: bool = False,
):
    """
    Unconstrained Huntington–Hill baseline (global allocation):
      - Start seats[i]=1 for each original OA-BAR leaf i
      - Allocate m_extra additional seats using HH priority based on WCRT weight
      - Realize k_i = seats[i]-1 extra splits inside leaf i

    ✅ Minimal fix:
      - supports OA-BAR 4/5/6-tuples
      - leaf_wcrt computed from original OA-BAR leaves
      - final_parts preserve provenance (meta + leaf_id)
    """
    n = len(parts)
    if m_extra < 0:
        raise ValueError("Require m_extra >= 0.")
    if n == 0:
        return ([], {"k": [], "seats": [], "leaf_wcrt": []}) if return_meta else []

    leaf_wcrt = []
    leaf_ids = []
    for i, part in enumerate(parts):
        r, o, *_rest = _unpack_part(part)
        leaf_wcrt.append(_safe_wcrt(Strip(r, o).calculate_region_wcrt()))
        # optional: keep leaf_id mapping if present
        _, _, _, _, _meta, lid = _unpack_part(part)
        leaf_ids.append(lid if lid is not None else i)

    seats = [1] * n

    for _ in range(m_extra):
        best_i = 0
        best_p = -1.0
        for i in range(n):
            p = _priority(leaf_wcrt[i], seats[i])
            if p > best_p:
                best_p = p
                best_i = i
        seats[best_i] += 1

    k = [s - 1 for s in seats]  # allocation vector: extra splits per original leaf

    final_parts = _apply_splits_per_leaf(
        parts,
        extra_splits_per_leaf=k,
        numerical_method=numerical_method,
        metric=metric,
    )

    if not return_meta:
        return final_parts

    # optional but helpful: k keyed by leaf_id too
    k_by_leaf_id = {leaf_ids[i]: int(k[i]) for i in range(n)}
    depth = (n - 1).bit_length()   # since n is power of two in OA-BAR experiments
    prefixes = [format(i, f"0{depth}b") for i in range(n)]

    alloc_meta: Dict[str, Any] = {
        "method": "HH",
        "n": n,
        "m_extra": int(m_extra),
        "leaf_wcrt": leaf_wcrt,
        "seats": seats,
        "k": k,                         # ✅ aligned with original OA-BAR order
        "prefixes": prefixes,  # ✅ NEW (pure metadata)
        "leaf_ids": leaf_ids,
        "k_by_leaf_id": k_by_leaf_id,   # ✅ optional, robust with your new leaf_id
    }
    return final_parts, alloc_meta
