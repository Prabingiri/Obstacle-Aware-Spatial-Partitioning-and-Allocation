# src/main_apportionment_final.py

import os
import json
import time
import copy
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import shape

from src.oabar.preprocessing import RegionWithObstacles
from src.oabar.hierarchical_decomposition_algorithm import HierarchicalDecomposition

from src.rba.baseline_greedy import greedy_maxfirst
from src.rba.baseline_huntington_hill import hh_apportion
from src.rba.apportionment_recursive_module_FINAL import apport_RBA as rba_apport

from src.rba.save_partitions_final_RBA import save_final_results
from src.rba.allocation_metrics import summarize_allocation


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _as4(part):
    """
    Accepts either:
      - 4-tuple: (reg, obs, seq, valid)
      - 5-tuple: (reg, obs, seq, valid, meta)
      - 6-tuple: (reg, obs, seq, valid, meta, leaf_id)
    Returns:
      - 4-tuple view: (reg, obs, seq, valid)
    """
    if len(part) < 4:
        raise ValueError(f"Bad partition tuple (len={len(part)}): {part}")
    return part[0], part[1], part[2], part[3]


def _copy_meta(part):
    """
    Returns meta dict if present, else {}.
    """
    if len(part) >= 5 and isinstance(part[4], dict):
        return dict(part[4])
    return {}


def _copy_leaf_id(part):
    """
    Returns leaf_id if present else None.
    """
    if len(part) >= 6:
        return part[5]
    return None


# -----------------------------
# 1) OA-BAR leaf construction
# -----------------------------
def build_oabar_leaves(
    region,
    obstacles,
    depth: int,
    numerical_method: str,
    metric: str,
) -> List[tuple]:
    """
    Returns OA-BAR leaves (list of tuples: (subregion, subobstacles, axis_seq, valid)).
    IMPORTANT: this must be called ONCE per (dataset, depth) so all methods share the same leaves.
    """
    prep = RegionWithObstacles(region, obstacles)
    decomp = HierarchicalDecomposition(
        prep.region,
        prep.get_simplified_obstacles(),
        max_depth=depth,
        metrics=metric,
        numerical_method=numerical_method,
        mode="track_back",
        allow_fallback_axis=True
    )
    leaves = decomp.run()
    return leaves


# -----------------------------
# 2) Run one method on fixed leaves
# -----------------------------
def run_one_method_on_fixed_leaves(
    leaves: List[tuple],
    method: str,
    m_extra: int,
    beta: float,
    numerical_method: str,
    metric: str,
) -> Tuple[List[tuple], float, Optional[Dict]]:
    """
    Returns (final_parts, runtime_seconds, alloc_meta_or_none).
    """
    method = method.upper().strip()

    leaves_copy = copy.deepcopy(leaves)
    # IMPORTANT: downstream apportionment methods may still expect 4-tuples.
    # We keep metadata in the tuple, but apportionment will read via helper unpackers.

    start = time.perf_counter()

    alloc_meta = None

    if method == "RBA":
        final_parts, alloc_meta = rba_apport(
            leaves_copy,
            m_extra=m_extra,
            beta=beta,
            numerical_method=numerical_method,
            metric=metric,
            return_meta=True,
        )

    elif method == "HH":
        final_parts, alloc_meta = hh_apportion(
            leaves_copy,
            m_extra=m_extra,
            beta=beta,
            numerical_method=numerical_method,
            metric=metric,
            return_meta=True,
        )

    elif method == "GREEDY":
        final_parts, alloc_meta = greedy_maxfirst(
            leaves_copy,
            m_extra=m_extra,
            numerical_method=numerical_method,
            metric=metric,
            return_meta=True,
        )

    else:
        raise ValueError(f"Unknown method={method}. Use RBA/HH/GREEDY.")

    elapsed = time.perf_counter() - start
    return final_parts, elapsed, alloc_meta



# -----------------------------
# 3) Experiment driver
# -----------------------------
def run_experiment_suite(
    dataset_files: Sequence[str],
    methods: Sequence[str],
    depths: Sequence[int],
    m_values: Sequence[int],
    *,
    beta: float = 1.0,
    numerical_method: str = "newton",
    metric: str = "NWCRT",
    output_root: str = "results/apport/experiments",
    # fairness knobs:
    enforce_rba_cap: bool = True,     # enforce m <= n-1 for RBA (recommended)
    enforce_all_cap: bool = True,    # if True, also cap HH/Greedy to m<=n-1
):
    """
    For each dataset file:
      for each depth:
        compute OA-BAR leaves ONCE
        for each m in m_values:
          for each method:
            run on same leaves and save outputs
    """
    import os
    import json
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = os.path.join(output_root, run_id)
    os.makedirs(run_root, exist_ok=True)

    logger.info(f"=== RUN ROOT: {run_root} ===")

    for fp in dataset_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        region = shape(data["region"])
        obstacles = [shape(o) for o in data["obstacles"]]
        meta = data.get("meta", {})

        # dataset_tag = os.path.splitext(os.path.basename(fp))[0]
        suite = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        dataset_tag = f"{suite}__{os.path.splitext(os.path.basename(fp))[0]}"
        pct = meta.get("obstacle_percentage", "NA")
        seed = meta.get("seed", "NA")
        logger.info(f"\n=== DATASET: {dataset_tag} | pct={pct} | seed={seed} ===")

        for depth in depths:
            leaves = build_oabar_leaves(region, obstacles, depth, numerical_method, metric)
            n = len(leaves)
            cap = n-1
            # m_grid = sorted({
            #     0,
            #     1,
            #     max(1, 10),
            #     max(1, 21),
            #     max(1, 43),
            #     max(1, 55),
            #     max(1, 77),
            #     max(1, 98),
            #     max (1, 119),
            #     cap,
            # })
            # m_grid_arbitrary = sorted(set(m_grid + [n, 2 * n, 4 * n]))
            # logger.info(f"OA-BAR built once: depth={depth} => n={n} leaves")

            def _round_pos(x: float) -> int:
                return max(1, int(round(x)))

            m_grid = sorted({
                0,
                _round_pos(0.05 * n),
                _round_pos(0.10 * n),
                _round_pos(0.20 * n),
                _round_pos(0.30 * n),
                _round_pos(0.40 * n),
                _round_pos(0.50 * n),
                _round_pos(0.60 * n),
                _round_pos(0.70 * n),
                cap,  # optional stress point; will be capped anyway if enforce_all_cap=True
            })

            m_grid_arbitrary = sorted(set(m_grid + [n, 2 * n, 4 * n]))
            # Decide valid m per depth based on constraints
            for m_extra in (m_grid if enforce_all_cap else m_grid_arbitrary):
                if m_extra < 0:
                    continue

                # cap logic
                cap = (n - 1) if n > 0 else 0

                for method in methods:
                    methodU = method.upper().strip()

                    if enforce_all_cap and m_extra > cap:
                        continue
                    if enforce_rba_cap and methodU == "RBA" and m_extra > cap:
                        continue

                    # run
                    final_parts, elapsed, alloc_meta = run_one_method_on_fixed_leaves(
                        leaves=leaves,
                        method=methodU,
                        m_extra=m_extra,
                        beta=beta,
                        numerical_method=numerical_method,
                        metric=metric,
                    )

                    logger.info(
                        f"[{dataset_tag}] depth={depth} m={m_extra} {methodU}: "
                        f"{len(final_parts)} parts | {elapsed:.2f}s"
                    )

                    # save
                    exp_tag = f"{dataset_tag}__d{depth}__m{m_extra}__{methodU}"
                    out_dir = os.path.join(run_root, dataset_tag, f"depth{depth}", f"m{m_extra}", methodU)

                    save_final_results(
                        partitions=final_parts,
                        datatype=exp_tag,
                        numerical_method=numerical_method,
                        user_metric=metric,
                        depth=depth,
                        output_dir=out_dir,
                        runtime=elapsed,
                        meta={
                            "run_root": run_root,
                            "dataset_tag": dataset_tag,
                            "method": methodU,
                            "depth": depth,
                            "n_leaves": n,
                            "m_extra": m_extra,
                            "total_parts": len(final_parts),
                            "obstacle_pct": meta.get("obstacle_percentage", "NA"),
                            "seed": meta.get("seed", "NA"),
                        },
                        write_wkt=False
                    )
                    # ---- allocation meta saving (minimal + robust) ----
                    import json, os

                    if alloc_meta is not None:
                        # standardize key (prefer k_extra)
                        if "k_extra" in alloc_meta:
                            kvec = alloc_meta["k_extra"]
                        elif "k" in alloc_meta:
                            kvec = alloc_meta["k"]
                            alloc_meta["k_extra"] = kvec  # normalize
                        else:
                            kvec = None

                        if kvec is not None:
                            alloc_meta["alloc_stats"] = summarize_allocation(kvec)

                        with open(os.path.join(out_dir, f"{exp_tag}_alloc.json"), "w", encoding="utf-8") as f:
                            json.dump(alloc_meta, f, indent=2)


# -----------------------------
# 4) Convenience: pick datasets from suite folder
# -----------------------------
def list_json_files(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            files.append(os.path.join(folder, name))
    return files


if __name__ == "__main__":

    # ---- point this to your generated suite folder ----
    # SUITE_DIR = "resource/dataset/synthetic_data/synthetic_data_generation/modified/RBA/cluster/cluster1_10percent"
    # dataset_files = "resource/dataset/synthetic_data/synthetic_data_generation/modified/RBA/cluster/data/synthetic_10pct_seed040.json"


    # SUITE_DIR = "resource/dataset/synthetic_data/synthetic_data_generation/modified/RBA/no_cluster_10"
    # dataset_files = list_json_files(SUITE_DIR)
    # CLUSTER_DIR = "resource/dataset/synthetic_data/synthetic_data_generation/modified/RBA/cluster2-final/data"
    UNIFORM_DIR = "resource/dataset/synthetic_data/synthetic_data_generation/modified/RBA/uniform/data"

    # dataset_files = list_json_files(CLUSTER_DIR) + list_json_files(UNIFORM_DIR)
    dataset_files = list_json_files(UNIFORM_DIR)



    # ---- methods to compare ----
    METHODS = ["RBA", "HH", "GREEDY"]

    # ---- depth grid (n = 2^depth) ----
    DEPTHS = [6, 7]   # 8, 16, 32 leaves
    # DEPTHS = [6]


    # ---- m grid ----
    # Recommended for clean plots: express m as fraction of n, but here we just give numbers.
    # We will automatically cap RBA at (n-1) unless you disable enforce_rba_cap.
    # M_VALUES = [1, 2, 4, 8, 12, 16, 24]  # will be filtered by caps per depth

    run_experiment_suite(
        dataset_files=dataset_files,
        methods=METHODS,
        depths=DEPTHS,
        m_values=[],
        beta=1.0,
        numerical_method="newton",
        metric="NWCRT",
        output_root="results/apport/experiments",
        enforce_rba_cap=True,
        enforce_all_cap=True,
    )
