# research_data_generator.py

import os
import random
import math
import json
import logging
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterSpec:
    """Optional clustered obstacle placement."""
    center: Tuple[float, float]   # (cx, cy)
    std_dev: float                # gaussian std-dev for x,y
    weight: float = 1.0           # mixture weight


class ResearchDataGenerator:
    """
    Research-grade synthetic obstacle generator.

    Key features:
      - Reproducible (seeded RNG)
      - Multiple obstacle coverages in one run
      - Uniform or clustered placement
      - Non-overlapping, fully-contained convex polygon obstacles
      - Connectivity safeguard (avoid disconnecting region)
      - Metadata saved with dataset for traceability
    """

    def __init__(
        self,
        region_size: Tuple[float, float] = (100.0, 100.0),
        obstacle_percentage: float = 5.0,
        max_obstacle_area: Optional[float] = 50.0,
        size_variation: float = 0.2,
        min_obstacle_area: float = 0.5,
        seed: Optional[int] = None,
        clusters: Optional[Sequence[Dict[str, Any]]] = None,
        max_attempts: int = 20000,
        fill_attempts: int = 20000,
        scaling_factor: float = 10.0,
        allow_touches: bool = False,   # False => no intersects/touches, True => allow touch but not overlap
    ):
        self.region_size = region_size
        self.obstacle_percentage = float(obstacle_percentage)

        self.max_obstacle_area = max_obstacle_area
        self.size_variation = float(size_variation)
        self.min_obstacle_area = float(min_obstacle_area)

        self.seed = seed
        self.max_attempts = int(max_attempts)
        self.fill_attempts = int(fill_attempts)
        self.scaling_factor = float(scaling_factor)

        self.allow_touches = bool(allow_touches)

        self._rng_py = random.Random(seed)
        self._rng_np = np.random.default_rng(seed)

        self.clusters: List[ClusterSpec] = []
        if clusters:
            for c in clusters:
                self.clusters.append(
                    ClusterSpec(
                        center=tuple(c["center"]),
                        std_dev=float(c["std_dev"]),
                        weight=float(c.get("weight", 1.0)),
                    )
                )

    # -----------------------
    # Geometry helpers
    # -----------------------

    def generate_region(self) -> Polygon:
        w, h = self.region_size
        return Polygon([(0, 0), (w, 0), (w, h), (0, h)])

    def _generate_convex_polygon(self, center: Tuple[float, float], radius: float, sides: int) -> Polygon:
        # random angles on circle, then convex hull
        angles = sorted(self._rng_py.uniform(0, 2 * math.pi) for _ in range(sides))
        pts = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]
        return Polygon(pts).convex_hull

    def _sample_center(self) -> Tuple[float, float]:
        w, h = self.region_size

        if not self.clusters:
            return (self._rng_py.uniform(0, w), self._rng_py.uniform(0, h))

        weights = np.array([c.weight for c in self.clusters], dtype=float)
        weights = weights / weights.sum()
        idx = self._rng_np.choice(len(self.clusters), p=weights)
        c = self.clusters[int(idx)]

        x = float(self._rng_np.normal(c.center[0], c.std_dev))
        y = float(self._rng_np.normal(c.center[1], c.std_dev))
        x = min(max(x, 0.0), w)
        y = min(max(y, 0.0), h)
        return (x, y)

    def _fits_and_nonoverlapping(self, region: Polygon, obs: Polygon, existing: List[Polygon]) -> bool:
        if not region.contains(obs):
            return False
        if not obs.is_valid:
            return False

        # Non-overlap constraint:
        # - if allow_touches=False: reject intersects() including touching
        # - if allow_touches=True: reject overlaps but allow boundary touch
        if not existing:
            return True

        if self.allow_touches:
            # "overlaps" misses containment; use intersection area > 0
            for e in existing:
                inter = obs.intersection(e)
                if not inter.is_empty and inter.area > 1e-12:
                    return False
            return True
        else:
            return all(not obs.intersects(e) for e in existing)

    # -----------------------
    # Main generation
    # -----------------------

    def generate_obstacles(self, region: Polygon) -> List[Polygon]:
        """
        Generate non-overlapping convex polygon obstacles whose UNION area
        is approximately obstacle_percentage% of region area.
        """
        target_area = (self.obstacle_percentage / 100.0) * region.area
        obstacles: List[Polygon] = []
        current_union_area = 0.0

        attempts = 0
        while current_union_area < target_area and attempts < self.max_attempts:
            remaining = max(target_area - current_union_area, 0.0)

            # pick an area scale based on remaining area
            base_area = remaining / max(self.scaling_factor, 1e-9)
            # convert area -> radius proxy (area ~ pi r^2)
            base_radius = math.sqrt(max(base_area, 0.0) / math.pi)

            # add variation
            jitter = 1.0 + self.size_variation * self._rng_py.uniform(-1.0, 1.0)
            radius = max(base_radius * jitter, math.sqrt(self.min_obstacle_area / math.pi))

            if self.max_obstacle_area is not None:
                radius = min(radius, math.sqrt(self.max_obstacle_area / math.pi))

            sides = self._rng_py.choice([3, 4, 5, 6])
            center = self._sample_center()
            candidate = self._generate_convex_polygon(center, radius, sides)

            if self._fits_and_nonoverlapping(region, candidate, obstacles):
                obstacles.append(candidate)
                current_union_area = unary_union(obstacles).area

            attempts += 1

        # fill phase with small obstacles if under target
        if current_union_area < target_area:
            logger.info("Fill phase: adding small obstacles to approach target coverage.")
            fixed_radius = math.sqrt(self.min_obstacle_area / math.pi)
            fill = 0
            while current_union_area < target_area and fill < self.fill_attempts:
                sides = self._rng_py.choice([3, 4, 5, 6])
                center = self._sample_center()
                candidate = self._generate_convex_polygon(center, fixed_radius, sides)

                if self._fits_and_nonoverlapping(region, candidate, obstacles):
                    obstacles.append(candidate)
                    current_union_area = unary_union(obstacles).area

                fill += 1

        if current_union_area < target_area:
            logger.warning(
                f"Target not reached (likely due to packing constraints). "
                f"Target={target_area:.2f}, Achieved={current_union_area:.2f}, #obs={len(obstacles)}"
            )

        return obstacles


    def ensure_connectivity(self, region: Polygon, obstacles: List[Polygon]) -> List[Polygon]:
        """
        Enforce connected free space: region \ union(obstacles) must be a single Polygon.
        If disconnected, iteratively drop obstacles (last-added first) until connected.
        """
        if not obstacles:
            return obstacles

        kept = list(obstacles)

        # quick check function
        def free_space_is_connected(obs_list: List[Polygon]) -> bool:
            u = unary_union(obs_list) if obs_list else None
            free = region.difference(u) if u else region
            return free.geom_type == "Polygon"  # single component only

        if free_space_is_connected(kept):
            return kept

        logger.warning("Free space disconnected; removing obstacles until connected.")
        # Conservative: pop obstacles until connected
        # (you can change order if you want “remove largest first”, but this is fine)
        while kept and not free_space_is_connected(kept):
            removed = kept.pop()
            logger.debug(f"Removed obstacle to restore connectivity: area={removed.area:.3f}")

        if not kept:
            logger.warning("Connectivity enforcement removed all obstacles (extreme packing).")

        return kept

    # -----------------------
    # IO helpers
    # -----------------------

    def generate_and_store(self, file_path: str) -> Dict[str, Any]:
        region = self.generate_region()
        obstacles = self.generate_obstacles(region)
        obstacles = self.ensure_connectivity(region, obstacles)

        union_poly = unary_union(obstacles) if obstacles else None
        total_area = float(union_poly.area) if union_poly else 0.0
        target_area = (self.obstacle_percentage / 100.0) * float(region.area)
        achieved_pct = (total_area / float(region.area) * 100.0) if region.area else 0.0

        meta = {
            "region_size": list(self.region_size),
            "obstacle_percentage": self.obstacle_percentage,
            "target_obstacle_area": target_area,
            "total_obstacle_area": total_area,
            "achieved_obstacle_percentage": achieved_pct,
            "num_obstacles": len(obstacles),
            "seed": self.seed,
            "generator_params": {
                "max_obstacle_area": self.max_obstacle_area,
                "min_obstacle_area": self.min_obstacle_area,
                "size_variation": self.size_variation,
                "scaling_factor": self.scaling_factor,
                "max_attempts": self.max_attempts,
                "fill_attempts": self.fill_attempts,
                "allow_touches": self.allow_touches,
                "clusters": [
                    {"center": list(c.center), "std_dev": c.std_dev, "weight": c.weight}
                    for c in self.clusters
                ],
            },
        }

        data = {
            "region": mapping(region),
            "obstacles": [mapping(o) for o in obstacles],
            "meta": meta,
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info(
            f"Saved synthetic dataset: {file_path} | "
            f"target={self.obstacle_percentage:.1f}% achieved={achieved_pct:.2f}% "
            f"obs={len(obstacles)}"
        )
        return data

    @staticmethod
    def load_from_file(file_path: str) -> Tuple[Polygon, List[Polygon], Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        region = shape(data["region"])
        obstacles = [shape(o) for o in data["obstacles"]]
        meta = data.get("meta", {})
        return region, obstacles, meta

    # -----------------------
    # Batch generation for experiments
    # -----------------------

    @staticmethod
    def generate_suite(
        out_dir: str,
        region_size: Tuple[float, float],
        obstacle_pcts: Sequence[int],
        seeds: Sequence[int],
        *,
        max_obstacle_area: float = 50.0,
        min_obstacle_area: float = 0.5,
        size_variation: float = 0.2,
        clusters: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Generate a dataset suite for experiments.
        Returns a list of file paths created.
        """
        paths: List[str] = []
        os.makedirs(out_dir, exist_ok=True)

        for pct in obstacle_pcts:
            for seed in seeds:
                gen = ResearchDataGenerator(
                    region_size=region_size,
                    obstacle_percentage=float(pct),
                    max_obstacle_area=max_obstacle_area,
                    size_variation=size_variation,
                    min_obstacle_area=min_obstacle_area,
                    seed=int(seed),
                    clusters=clusters,
                )
                fp = os.path.join(out_dir, f"synthetic_{pct:02d}pct_seed{seed:03d}.json")
                gen.generate_and_store(fp)
                paths.append(fp)

        return paths


# -----------------------
# Example usage (safe defaults)
# -----------------------
if __name__ == "__main__":
    OBSTACLE_PCTS = [5, 10, 20, 30, 40, 50]
    SEEDS = [40]

    clusters1 = [
        {"center": (30, 30), "std_dev": 70, "weight": 0.25},
        {"center": (70, 70), "std_dev": 50, "weight": 0.25},
    ]
    clusters2 = [
        {"center": (25, 25), "std_dev": 6, "weight": 0.25},
        {"center": (50, 50), "std_dev": 6, "weight": 0.50},
        {"center": (75, 75), "std_dev": 6, "weight": 0.25},
    ]

    out_dir = "synthetic_data_generation/modified/RBA/cluster2-final/data"
    ResearchDataGenerator.generate_suite(
        out_dir=out_dir,
        region_size=(100, 100),
        obstacle_pcts=OBSTACLE_PCTS,
        seeds=SEEDS,
        max_obstacle_area=250.0,
        min_obstacle_area=5.0,
        size_variation=0.8,
        clusters=clusters1,  # set clusters to enable non-uniform placement
    )

    # sanity check
    sample_fp = os.path.join(out_dir, "synthetic_05pct_seed001.json")
    region, obstacles, meta = ResearchDataGenerator.load_from_file(sample_fp)
    logger.info(f"Loaded {sample_fp}: {meta.get('achieved_obstacle_percentage', 'NA')}% coverage, "
                f"{meta.get('num_obstacles', 'NA')} obstacles")
