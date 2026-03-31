# src/apportionment.py  (or src/apportionment_recursive_module_FINAL.py)
# DROP-IN replacement with minimal changes:
#  (1) supports OA-BAR 4/5/6-tuples (preserve provenance meta + leaf_id)
#  (2) add _safe_wcrt and use it when computing leaf_wcrt
#  (3) call apply_extra_splits with keyword args
#  (4) include "k" in alloc_meta (standardize with HH/GREEDY); keep k_extra too

import math
import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Any, Optional

from src.oabar.strip_perimeter import Strip
from src.oabar.optimal_axis_selection import OptimalAxisSelection

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(format="%(message)s", level=logging.INFO)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _safe_wcrt(w: float) -> float:
    # Prevent NaN/inf/negative from breaking priority ordering
    if w is None or not math.isfinite(w) or w < 0:
        return 0.0
    return float(w)


# =========================
# ✅ NEW (MINIMAL): tuple compatibility helpers
# =========================
def _unpack_part(part):
    """
    Supports:
      - 4-tuple: (reg, obs, seq, valid)
      - 5-tuple: (reg, obs, seq, valid, meta)
      - 6-tuple: (reg, obs, seq, valid, meta, leaf_id)
    Returns:
      reg, obs, seq, valid, meta_or_none, leaf_id_or_none
    """
    if part is None or len(part) < 4:
        raise ValueError(f"Bad partition tuple (len={len(part) if part is not None else 'None'}): {part}")
    reg, obs, seq, valid = part[0], part[1], part[2], part[3]
    meta = part[4] if len(part) >= 5 and isinstance(part[4], dict) else None
    leaf_id = part[5] if len(part) >= 6 else None
    return reg, obs, seq, valid, meta, leaf_id


def _repack_part(reg, obs, seq, valid, meta, leaf_id):
    """
    Preserve 4 vs 5 vs 6 tuple form.
    If meta/leaf_id absent -> 4-tuple. If meta present -> 5/6 tuple.
    """
    if meta is None and leaf_id is None:
        return (reg, obs, seq, valid)
    if leaf_id is None:
        return (reg, obs, seq, valid, meta)
    return (reg, obs, seq, valid, meta, leaf_id)


def _validate_prefixes(prefixes: List[str]) -> int:
    if not prefixes:
        raise ValueError("prefixes must be non-empty.")
    x = len(prefixes[0])
    if any(len(p) != x for p in prefixes):
        raise ValueError("All prefixes must have the same bit-length.")
    if any(any(ch not in "01" for ch in p) for p in prefixes):
        raise ValueError("prefixes must be binary strings.")
    return x


def _phi(u: float, beta: float) -> float:
    u = max(0.0, float(u))
    if beta == 1.0:
        return u
    return u ** beta


def _crowd_tree(addr: str, c: DefaultDict[str, int]) -> float:
    x = len(addr)
    s = 0.0
    for t in range(1, x + 1):
        p = addr[:t - 1]
        b = addr[t - 1]
        sib = p + ("1" if b == "0" else "0")
        s += (2.0 ** (-t)) * float(c[sib])
    return s


def _feasible_prefix(addr: str, c: DefaultDict[str, int], delta: int = 1) -> bool:
    x = len(addr)
    for t in range(0, x):
        p = addr[:t]
        b = addr[t]
        p_in = p + b
        p_sib = p + ("1" if b == "0" else "0")
        if abs((c[p_in] + 1) - c[p_sib]) > delta:
            return False
    return True


def _accept_prefix(addr: str, c: DefaultDict[str, int]) -> None:
    x = len(addr)
    for t in range(0, x):
        c[addr[:t + 1]] += 1


def rba_allocate(
    U: List[float],
    prefixes: List[str],
    m: int,
    beta: float = 1.0,
    delta: int = 1,
    q_round_decimals: int = 6,
) -> Tuple[List[int], List[int]]:
    n = len(U)
    if n != len(prefixes):
        raise ValueError("U and prefixes length mismatch.")
    if not (0 <= m < n):
        raise ValueError("Require 0 <= m < n.")
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    if delta < 0:
        raise ValueError("delta must be >= 0.")

    _ = _validate_prefixes(prefixes)

    k = [1] * n
    chosen = [False] * n
    order: List[int] = []

    c: DefaultDict[str, int] = defaultdict(int)

    Q = [round(_phi(U[i], beta), q_round_decimals) for i in range(n)]

    seats_left = m
    while seats_left > 0:
        eligible = [i for i in range(n) if not chosen[i]]
        if not eligible:
            raise RuntimeError("All leaves selected but seats remain.")

        blocked = set()

        while True:
            remaining = [i for i in eligible if i not in blocked]
            if not remaining:
                raise RuntimeError("RBA invariant violated: no feasible leaf remains.")

            best = max(Q[i] for i in remaining)
            tier = [i for i in remaining if Q[i] == best]

            # Deterministic tie-break: lower CrowdTree, then lower index
            tier.sort(key=lambda i: (_crowd_tree(prefixes[i], c), i))

            accepted = False
            for i in tier:
                if _feasible_prefix(prefixes[i], c, delta=delta):
                    chosen[i] = True
                    k[i] = 2
                    _accept_prefix(prefixes[i], c)
                    order.append(i + 1)  # 1-based
                    seats_left -= 1
                    accepted = True
                    break
                else:
                    blocked.add(i)

            if accepted:
                break

    return k, order


def apply_extra_splits(parts, k, method: str = "brent", metric: str = "NWCRT"):
    """
    ✅ MINIMAL FIX:
      - supports 4/5/6 tuple leaves
      - preserves meta + leaf_id on children (so plotting can reconstruct ancestry)
    """
    out = []
    for part, ki in zip(parts, k):
        reg, obs, seq, valid, meta, leaf_id = _unpack_part(part)

        if ki == 1:
            out.append(_repack_part(reg, obs, seq, valid, meta, leaf_id))
        else:
            seq0 = [] if seq is None else list(seq)
            sel = OptimalAxisSelection(reg, obs, metric, method)
            axis, _, _, (Lr, Lo), (Rr, Ro) = sel.select_best_axis()

            # Preserve provenance (meta + leaf_id stay attached to this original OA-BAR leaf)
            out.append(_repack_part(Lr, Lo, seq0 + [axis], True, meta, leaf_id))
            out.append(_repack_part(Rr, Ro, seq0 + [axis], True, meta, leaf_id))

    return out


def apport_RBA(
    initial_parts,
    m_extra: int,
    beta: float,
    numerical_method: str = "brent",
    metric: str = "NWCRT",
    return_meta: bool = False,
):
    """
    Phase-1 RBA (delta=1, single-step):
      - n must be power-of-two (OA-BAR depth)
      - 0 <= m_extra < n  (at most one extra split per original leaf)
      - returns final_parts, and optionally alloc metadata.
    """
    n = len(initial_parts)
    if n == 0:
        empty = {"k": [], "k_extra": [], "seats": [], "leaf_wcrt": []}
        return ([], empty) if return_meta else []
    if not _is_power_of_two(n):
        raise ValueError(f"RBA expects n to be a power of two (got n={n}).")
    if not (0 <= m_extra < n):
        raise ValueError("Require 0 <= m_extra < n.")
    if beta <= 0:
        raise ValueError("beta must be > 0.")

    depth = (n - 1).bit_length()

    # ✅ MINIMAL FIX: compute leaf_wcrt safely on 4/5/6 tuples
    leaf_wcrt = []
    for part in initial_parts:
        r, o, _, _, _, _ = _unpack_part(part)
        leaf_wcrt.append(_safe_wcrt(Strip(r, o).calculate_region_wcrt()))

    prefixes = [format(i, f"0{depth}b") for i in range(n)]

    seats, order = rba_allocate(
        leaf_wcrt,
        prefixes,
        m_extra,
        beta=beta,
        delta=1,
        q_round_decimals=6,
    )

    final_parts = apply_extra_splits(
        initial_parts,
        seats,
        method=numerical_method,
        metric=metric,
    )

    if not return_meta:
        return final_parts

    k_extra = [s - 1 for s in seats]  # 0/1 vector for phase-1

    alloc_meta: Dict[str, Any] = {
        "method": "RBA",
        "phase": "single_step_delta1",
        "n": n,
        "m_extra": int(m_extra),
        "beta": float(beta),
        "delta": 1,
        "leaf_wcrt": leaf_wcrt,
        "prefixes": prefixes,
        "seats": seats,        # in {1,2}
        "k": k_extra,          # standardized key
        "k_extra": k_extra,    # keep for clarity/back-compat
        "order": order,        # 1-based acceptance order
    }
    return final_parts, alloc_meta
