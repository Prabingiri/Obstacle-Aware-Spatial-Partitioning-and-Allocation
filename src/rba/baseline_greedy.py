# src/greedy_baseline.py
from typing import Dict, List, Tuple, Any, Optional
import math

from src.oabar.strip_perimeter import Strip
from src.oabar.optimal_axis_selection import OptimalAxisSelection


def _safe_wcrt(w: float) -> float:
    if w is None or not math.isfinite(w) or w < 0:
        return 0.0
    return float(w)


# ✅ MINIMAL: tuple compatibility helpers (same idea as in RBA)
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


def greedy_maxfirst(
    parts,
    m_extra: int,
    numerical_method="brent",
    metric="NWCRT",
    return_meta: bool = False,
):
    """
    Greedy baseline:
      repeat m_extra times:
        pick current leaf with max WCRT
        split it once

    Allocation vector over original OA-BAR leaves:
      k[i] = how many times greedy selected (a descendant of) original leaf i to split.

    ✅ Minimal fix:
      - supports OA-BAR 4/5/6-tuples
      - preserves provenance (meta + leaf_id)
      - origin tracking uses leaf_id if available, else initial index
    """
    # Attach origin index + leaf_id to each current leaf
    # Each cur item: (reg, obs, seq, valid, meta, leaf_id, origin_index, origin_leaf_id)
    cur = []
    for i, part in enumerate(parts):
        reg, obs, seq, valid, meta, leaf_id = _unpack_part(part)
        origin_index = i
        origin_leaf_id = leaf_id if leaf_id is not None else i
        cur.append((reg, obs, seq, valid, meta, leaf_id, origin_index, origin_leaf_id))

    n = len(cur)
    k = [0] * n                      # by original index order (stable)
    k_by_leaf_id: Dict[Any, int] = {}  # extra safety: by leaf_id (if you want it later)
    depth = (n - 1).bit_length()
    prefixes = [format(i, f"0{depth}b") for i in range(n)]
    leaf_ids = []
    for i, part in enumerate(parts):
        *_r, _o, _s, _v, _meta, lid = _unpack_part(part)
        leaf_ids.append(lid if lid is not None else i)

    for _ in range(m_extra):
        if not cur:
            break

        wcrts = [_safe_wcrt(Strip(r, o).calculate_region_wcrt()) for (r, o, *_rest) in cur]
        idx = max(range(len(cur)), key=lambda j: wcrts[j])

        reg, obs, seq, valid, meta, leaf_id, origin_index, origin_leaf_id = cur.pop(idx)
        k[origin_index] += 1
        k_by_leaf_id[origin_leaf_id] = k_by_leaf_id.get(origin_leaf_id, 0) + 1

        seq0 = [] if seq is None else list(seq)
        sel = OptimalAxisSelection(reg, obs, metric, numerical_method)
        axis, _, _, (Lr, Lo), (Rr, Ro) = sel.select_best_axis()

        # Children inherit the same origin + provenance (they came from that OA-BAR base leaf)
        cur.append((Lr, Lo, seq0 + [axis], True, meta, leaf_id, origin_index, origin_leaf_id))
        cur.append((Rr, Ro, seq0 + [axis], True, meta, leaf_id, origin_index, origin_leaf_id))

    # ✅ Preserve provenance in final parts (so plotting can reconstruct subtrees)
    final_parts = [_repack_part(r, o, s, v, meta, leaf_id) for (r, o, s, v, meta, leaf_id, *_rest) in cur]

    if not return_meta:
        return final_parts

    # leaf_wcrt on original OA-BAR leaves (parts)
    leaf_wcrt = []
    for part in parts:
        r, o, *_ = _unpack_part(part)
        leaf_wcrt.append(_safe_wcrt(Strip(r, o).calculate_region_wcrt()))

    alloc_meta: Dict = {
        "method": "GREEDY",
        "n": n,
        "m_extra": int(m_extra),
        "leaf_wcrt": leaf_wcrt,
        "k": k,                         # ✅ aligned with original OA-BAR order
        "prefixes": prefixes,  # ✅ NEW (pure metadata)
        "leaf_ids": leaf_ids,
        "k_by_leaf_id": k_by_leaf_id,   # ✅ optional but useful with your new leaf_id
    }
    return final_parts, alloc_meta
