# src/allocation_metrics.py
import math
from typing import List, Dict


def _safe(v: float) -> float:
    if v is None or not math.isfinite(v) or v < 0:
        return 0.0
    return float(v)


def jain_fairness(k: List[float], eps: float = 1e-12) -> float:
    x = [_safe(v) for v in k]
    s1 = sum(x)
    s2 = sum(v * v for v in x)
    if s2 < eps:
        return 1.0
    return (s1 * s1) / (len(x) * s2)


def entropy(k: List[float], eps: float = 1e-12) -> float:
    x = [_safe(v) for v in k]
    s = sum(x)
    if s < eps:
        return 0.0
    p = [v / s for v in x if v > 0]
    return -sum(pi * math.log(pi + eps) for pi in p)


def gini(k: List[float], eps: float = 1e-12) -> float:
    x = sorted([_safe(v) for v in k])
    n = len(x)
    s = sum(x)
    if s < eps:
        return 0.0
    # classic Gini formula
    cum = 0.0
    for i, xi in enumerate(x, start=1):
        cum += i * xi
    return (2 * cum) / (n * s) - (n + 1) / n


def summarize_allocation(k: List[float]) -> Dict[str, float]:
    return {
        "gini": gini(k),
        "jain": jain_fairness(k),
        "entropy": entropy(k),
        "k_max": max(k) if k else 0.0,
        "k_sum": sum(k) if k else 0.0,
    }
