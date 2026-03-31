import csv
import datetime
import json
import logging
import os
import re
import statistics
from typing import Any, Dict, Optional

from src.oabar.strip_perimeter import Strip

logger = logging.getLogger(__name__)


def _percentile(xs, q: float) -> float:
    """q in [0,1]. simple deterministic percentile (no numpy dependency)."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[min(max(idx, 0), len(xs) - 1)])


def _jain_fairness(xs) -> float:
    if not xs:
        return 1.0
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    n = len(xs)
    if s2 <= 1e-12:
        return 1.0
    return (s1 * s1) / (n * s2)


# =========================
# ✅ NEW (MINIMAL): tuple compatibility helper
# =========================
def _unpack_part(part):
    """
    Supports:
      - 4-tuple: (subreg, subobs, axis_seq, valid)
      - 5-tuple: (subreg, subobs, axis_seq, valid, meta)
      - 6-tuple: (subreg, subobs, axis_seq, valid, meta, leaf_id)
    Returns:
      subreg, subobs, axis_seq, valid, meta_or_none, leaf_id_or_none
    """
    if part is None or len(part) < 4:
        raise ValueError(f"Bad partition tuple (len={len(part) if part is not None else 'None'}): {part}")
    subreg, subobs, axis_seq, valid = part[0], part[1], part[2], part[3]
    meta = part[4] if len(part) >= 5 and isinstance(part[4], dict) else None
    leaf_id = part[5] if len(part) >= 6 else None
    return subreg, subobs, axis_seq, valid, meta, leaf_id


def save_final_results(
    partitions,
    datatype,
    numerical_method,
    user_metric,
    depth,
    output_dir="results/obstacle_aware",
    runtime=None,
    *,
    # NEW: structured metadata (pass from main driver when possible)
    meta: Optional[Dict[str, Any]] = None,
    write_wkt: bool = False,   # NEW: WKT makes CSV huge; keep False for suites
):
    """
    Saves per-partition rows + one Overall row with strong aggregation metrics.
    Also updates a robust JSON summary keyed by dataset+method+params (NOT by fragile string parsing).
    """

    os.makedirs(output_dir, exist_ok=True)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    import hashlib

    def _short(s: str, maxlen: int = 40) -> str:
        s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
        return s[:maxlen]

    tag_short = _short(datatype, 45)
    sig = hashlib.md5(f"{datatype}_{numerical_method}_{user_metric}_{depth}_{now_str}".encode("utf-8")).hexdigest()[:8]
    base_filename = f"{tag_short}__{sig}"

    # base_filename = f"{datatype}_{numerical_method}_{user_metric}_depth{depth}_{now_str}"

    csv_filename = os.path.join(output_dir, f"{base_filename}.csv")
    tree_filename = os.path.join(output_dir, f"{base_filename}_tree.txt")
    visualization_filename = os.path.join(output_dir, f"{base_filename}_partition.png")

    # ---- robust summary file (do not hardcode AR/obstacle-aware paths) ----
    meta = meta or {}
    run_root = meta.get("run_root", output_dir)  # fallback: old behavior still works
    summary_file = os.path.join(run_root, "apportionment_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    summary_data = {}
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    meta = meta or {}
    # expected meta fields (best-effort):
    # dataset_tag, obstacle_pct, seed, method, n_leaves, m_extra, total_parts
    dataset_tag = meta.get("dataset_tag", "unknown_dataset")
    method = meta.get("method", "UNKNOWN")
    m_extra = meta.get("m_extra", "NA")
    n_leaves = meta.get("n_leaves", "NA")
    obstacle_pct = meta.get("obstacle_pct", "NA")
    seed = meta.get("seed", "NA")
    total_parts = meta.get("total_parts", len(partitions))

    # Unique key per experiment setting
    exp_key = f"{dataset_tag}__{method}__d{depth}__m{m_extra}"

    # ---- CSV columns ----
    columns = [
        # experiment identifiers (NEW, critical)
        "dataset_tag", "method", "depth", "n_leaves", "m_extra", "total_parts",
        "obstacle_pct", "seed",
        # per-part row identifiers
        "partition_number", "num_obstacles",
        # per-part metrics
        "WCRT", "aspect_ratio",
        "sequence_of_chosen_axes",
        # overall metrics (filled only for Overall row)
        "min_wcrt", "max_wcrt", "mean_wcrt", "p95_wcrt",
        "std_wcrt", "cv_wcrt", "range_wcrt", "jain_wcrt",
        "mean_ar", "min_ar",
        "runtime",
        # optional heavy fields
        "partition_boundary_wkt", "obstacles_bounds",
        # ✅ optional provenance (lightweight, does not bloat)
        "leaf_id", "oabar_addr",
    ]

    wcrt_values = []
    aspect_ratios = []
    num_obstacles_list = []

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        # ✅ MINIMAL: use _unpack_part; ignore meta unless present
        for i, part in enumerate(partitions, start=1):
            subreg, subobs, axis_seq, valid, pmeta, leaf_id = _unpack_part(part)

            wcrt_val = Strip(subreg, subobs).calculate_region_wcrt()
            ar_val = compute_aspect_ratio(subreg)

            wcrt_values.append(float(wcrt_val))
            aspect_ratios.append(float(ar_val))
            num_obstacles_list.append(len(subobs))

            # optional provenance
            oabar_addr = ""
            if isinstance(pmeta, dict):
                oabar_addr = str(pmeta.get("addr", ""))

            row = {
                "dataset_tag": dataset_tag,
                "method": method,
                "depth": depth,
                "n_leaves": n_leaves,
                "m_extra": m_extra,
                "total_parts": total_parts,
                "obstacle_pct": obstacle_pct,
                "seed": seed,

                "partition_number": i,
                "num_obstacles": len(subobs),

                "WCRT": round(wcrt_val, 6),
                "aspect_ratio": round(ar_val, 6),
                "sequence_of_chosen_axes": "->".join(axis_seq) if axis_seq else "",

                "runtime": "",
                "partition_boundary_wkt": subreg.wkt if write_wkt else "",
                "obstacles_bounds": ";".join([str(o.bounds) for o in subobs]),

                # provenance (optional)
                "leaf_id": leaf_id if leaf_id is not None else "",
                "oabar_addr": oabar_addr,
            }
            writer.writerow(row)

        # ---- overall metrics ----
        if wcrt_values:
            min_wcrt = min(wcrt_values)
            max_wcrt = max(wcrt_values)
            mean_wcrt = sum(wcrt_values) / len(wcrt_values)
            std_wcrt = statistics.pstdev(wcrt_values) if len(wcrt_values) > 1 else 0.0
            cv_wcrt = (std_wcrt / mean_wcrt) if mean_wcrt > 1e-12 else 0.0
            p95_wcrt = _percentile(wcrt_values, 0.95)
            range_wcrt = max_wcrt - min_wcrt
            jain = _jain_fairness(wcrt_values)

            mean_ar = statistics.mean(aspect_ratios) if aspect_ratios else 0.0
            min_ar = min(aspect_ratios) if aspect_ratios else 0.0
        else:
            min_wcrt = max_wcrt = mean_wcrt = std_wcrt = cv_wcrt = p95_wcrt = range_wcrt = 0.0
            jain = 1.0
            mean_ar = min_ar = 0.0

        writer.writerow({
            "dataset_tag": dataset_tag,
            "method": method,
            "depth": depth,
            "n_leaves": n_leaves,
            "m_extra": m_extra,
            "total_parts": total_parts,
            "obstacle_pct": obstacle_pct,
            "seed": seed,

            "partition_number": "Overall",
            "num_obstacles": statistics.mean(num_obstacles_list) if num_obstacles_list else 0,

            "WCRT": "",
            "aspect_ratio": "",
            "sequence_of_chosen_axes": "",

            "min_wcrt": round(min_wcrt, 6),
            "max_wcrt": round(max_wcrt, 6),
            "mean_wcrt": round(mean_wcrt, 6),
            "p95_wcrt": round(p95_wcrt, 6),

            "std_wcrt": round(std_wcrt, 6),
            "cv_wcrt": round(cv_wcrt, 6),
            "range_wcrt": round(range_wcrt, 6),
            "jain_wcrt": round(jain, 6),

            "mean_ar": round(mean_ar, 6),
            "min_ar": round(min_ar, 6),

            "runtime": runtime if runtime is not None else "",
            "partition_boundary_wkt": "",
            "obstacles_bounds": "",

            # provenance empty for overall row
            "leaf_id": "",
            "oabar_addr": "",
        })

    logger.info(f"[save_final_results] CSV saved to {csv_filename}")
    os.makedirs(os.path.dirname(tree_filename), exist_ok=True)
    # ---- tree file (unchanged logic, but tuple-safe) ----
    with open(tree_filename, "w", encoding="utf-8") as tf:
        for i, part in enumerate(partitions, start=1):
            subreg, subobs, axis_seq, validity, _pmeta, _leaf_id = _unpack_part(part)
            depth_level = len(axis_seq) if axis_seq else 0
            indent = "  " * depth_level
            line = (
                f"{indent}- Partition {i} (Depth {depth_level}): axes=({('->'.join(axis_seq) if axis_seq else '')}) "
                f"BBox={subreg.bounds}, #Obstacles={len(subobs)}"
            )
            tf.write(line + "\n")

    # ---- summary JSON (robust) ----
    summary_data[exp_key] = {
        "dataset_tag": dataset_tag,
        "method": method,
        "depth": depth,
        "n_leaves": n_leaves,
        "m_extra": m_extra,
        "total_parts": total_parts,
        "obstacle_pct": obstacle_pct,
        "seed": seed,

        "min_wcrt": round(min_wcrt, 6),
        "max_wcrt": round(max_wcrt, 6),
        "mean_wcrt": round(mean_wcrt, 6),
        "p95_wcrt": round(p95_wcrt, 6),
        "std_wcrt": round(std_wcrt, 6),
        "cv_wcrt": round(cv_wcrt, 6),
        "jain_wcrt": round(jain, 6),

        "mean_ar": round(mean_ar, 6),
        "min_ar": round(min_ar, 6),

        "runtime": runtime if runtime is not None else None,
        "csv": csv_filename,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"[save_final_results] Summary updated: {summary_file}")

    # ---- visualization (keep your existing function) ----
    try:
        save_partition_visualization(partitions, filename=visualization_filename,
                                     show_wcrt=True, show_obstacles=True)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def compute_aspect_ratio(geometry, eps=1e-9):
    """
    Compute squareness as the minimum of w/h and h/w.
    """
    minx, miny, maxx, maxy = geometry.bounds
    w = maxx - minx
    h = maxy - miny
    if abs(w) < eps or abs(h) < eps:
        return 1.0  # Degenerate shapes are treated as perfectly square
    return min(w / h, h / w)


def save_partition_visualization(partitions, filename, show_obstacles=True, show_wcrt=True):
    """
    Visualizes final partitions using matplotlib and saves the figure to a file.

    Parameters:
        partitions : list
            List of partitions, each containing subregion, subobstacles, axis sequence, validity.
        filename : str
            Path where the visualization image will be saved.
        show_obstacles : bool, optional
            Whether to draw obstacles within each partition.
        show_wcrt : bool, optional
            Whether to display WCRT values on each partition.
    """
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon
    import random

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = []

    # Generate random colors for partitions
    for _ in range(len(partitions)):
        c = (random.random(), random.random(), random.random())
        colors.append(c)

    # ✅ MINIMAL: use _unpack_part; ignore provenance unless needed
    for i, part in enumerate(partitions):
        subregion, subobs, ax_seq, validity, _pmeta, _leaf_id = _unpack_part(part)
        color = colors[i]
        # Draw subregion
        plot_polygon(subregion, ax=ax, add_points=False,
                     facecolor=color, alpha=0.4, edgecolor='black')

        label_txt = f"Partition {i + 1}"
        # Calculate and display WCRT if required
        if show_wcrt:
            from src.strip_perimeter import Strip  # Ensure Strip is imported
            strip_mgr = Strip(subregion, subobs)
            wcrt_val = strip_mgr.calculate_region_wcrt()
            label_txt += f"\nWCRT={wcrt_val:.2f}"

        # Place text label at the centroid of the subregion
        cx, cy = subregion.centroid.x, subregion.centroid.y
        ax.text(cx, cy, label_txt, ha='center', va='center',
                fontsize=8, color='black')

        # Draw obstacles if required
        if show_obstacles:
            for ob in subobs:
                plot_polygon(ob, ax=ax, facecolor='none', edgecolor='red')

    ax.set_title("Final Partitions")
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(filename)  # Save the figure to the given filename
    plt.close(fig)  # Close the figure to free memory
