import logging
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from src.oabar.optimal_axis_selection import OptimalAxisSelection

#####################
DRONE_THRESHOLD = 5.0
COVERAGE_RATIO_STOP = 0.90
MIN_LARGEST_CONNECTED_SPACE = DRONE_THRESHOLD
#####################


class HierarchicalDecomposition:
    """
    DROP-IN replacement (minimal additive changes):
      - Adds deterministic OA-BAR leaf identity / ancestry metadata.
      - No behavior change in splitting logic.
      - Leaves now store provenance: leaf_id + binary address (left/right path).
    """

    def __init__(
        self,
        region,
        obstacles,
        max_depth=3,
        metrics="TotalWCRT",
        numerical_method="newton",
        min_dimension_threshold=1e-3,
        check_connectivity=False,
        allow_fallback_axis=True,
        mode="track_back",
    ):
        self.region = self._validate_geometry(region)

        # --- Minimal robustness: ensure obstacles are valid AND clipped to region ---
        obs_clean = []
        for obs in (obstacles or []):
            o = self._validate_geometry(obs)
            if o is None or o.is_empty:
                continue
            try:
                o = o.intersection(self.region)
            except Exception:
                pass
            if o is not None and (not o.is_empty):
                obs_clean.append(o)
        self.obstacles = obs_clean

        self.max_depth = max_depth
        self.metrics = metrics
        self.numerical_method = numerical_method
        self.min_dimension_threshold = min_dimension_threshold
        self.check_connectivity = check_connectivity
        self.allow_fallback_axis = allow_fallback_axis
        self.mode = mode

        # Leaves are stored here
        self.partitions = []

        # Existing: axis sequence stack (x/y)
        self.axis_stack = []

        # ✅ NEW: L/R address bits stack (0=left, 1=right)
        self._addr_bits_stack = []

        # ✅ NEW: deterministic leaf id assigned in left-to-right DFS order
        self._leaf_id_counter = 0

        # ✅ NEW: internal node id counter (optional but useful)
        self._node_id_counter = 0

        # --- Numerical safety epsilon (used only where needed) ---
        self._EPS = 1e-9

        # logger
        self._log = logging.getLogger(__name__)

    def _validate_geometry(self, geom):
        if geom is None:
            return geom
        if not geom.is_valid:
            geom = make_valid(geom)
        return geom

    # ✅ NEW: helper to snapshot per-leaf metadata
    def _make_leaf_meta(self, region, depth: int, parent_id: int):
        minx, miny, maxx, maxy = region.bounds
        addr = "".join(self._addr_bits_stack)  # binary path string
        leaf_id = self._leaf_id_counter
        meta = {
            "leaf_id": leaf_id,
            "addr": addr,              # '' at root, then '0','1','00',...
            "depth": depth,
            "parent_id": parent_id,
            "bbox": (float(minx), float(miny), float(maxx), float(maxy)),
        }
        return meta, leaf_id

    def run(self):
        # root has no parent
        produced_any = self._decompose(self.region, self.obstacles, depth=0, parent_id=-1)

        if self.mode == "track_back" and not produced_any:
            # keep original behavior, but include provenance too
            meta, leaf_id = self._make_leaf_meta(self.region, depth=0, parent_id=-1)
            self._leaf_id_counter += 1
            self.partitions.append((self.region, self.obstacles, list(self.axis_stack), True, meta, leaf_id))

        return self.partitions

    # ✅ SIGNATURE CHANGE: add parent_id (additive)
    def _decompose(self, region, obstacles, depth, parent_id: int) -> bool:
        if region is None or region.is_empty:
            return False

        # Give this region an internal node id (useful for tracing)
        node_id = self._node_id_counter
        self._node_id_counter += 1

        region_area = region.area
        obs_area = sum(ob.area for ob in obstacles) if obstacles else 0.0
        coverage_ratio = (obs_area / region_area) if region_area > 1e-12 else 1.0
        free_area = region_area - obs_area

        largest_hole_area = self._compute_largest_free_space(region, obstacles)

        # coverage stop
        if coverage_ratio >= COVERAGE_RATIO_STOP:
            if free_area < DRONE_THRESHOLD and largest_hole_area < DRONE_THRESHOLD:
                is_ok = self._is_subregion_valid(region, obstacles, largest_hole=largest_hole_area)

                meta, leaf_id = self._make_leaf_meta(region, depth=depth, parent_id=parent_id)
                self._leaf_id_counter += 1

                self.partitions.append((region, obstacles, list(self.axis_stack), is_ok, meta, leaf_id))
                return True  # produced a leaf

        # max depth stop
        if depth >= self.max_depth:
            is_ok = self._is_subregion_valid(region, obstacles, largest_hole=largest_hole_area)

            meta, leaf_id = self._make_leaf_meta(region, depth=depth, parent_id=parent_id)
            self._leaf_id_counter += 1

            self.partitions.append((region, obstacles, list(self.axis_stack), is_ok, meta, leaf_id))
            return True  # produced a leaf (important!)

        # validity check before splitting (use cached largest_hole_area)
        if not self._is_subregion_valid(region, obstacles, largest_hole=largest_hole_area):
            return False

        # axis selection (compute BOTH axes once)
        try:
            axis_selector = OptimalAxisSelection(
                region, obstacles,
                user_metric=self.metrics,
                numerical_method=self.numerical_method
            )
            best_axis, overall_metrics, best_div_pt, subL, subR = axis_selector.select_best_axis()
        except Exception:
            # --- True track-back at node: store last valid region instead of losing it ---
            is_ok = self._is_subregion_valid(region, obstacles, largest_hole=largest_hole_area)

            meta, leaf_id = self._make_leaf_meta(region, depth=depth, parent_id=parent_id)
            self._leaf_id_counter += 1

            self.partitions.append((region, obstacles, list(self.axis_stack), is_ok, meta, leaf_id))
            return True

        # try best axis
        if self._attempt_partition(region, obstacles, best_axis, depth, best_div_pt, subL, subR, parent_id=node_id):
            return True

        # true fallback: FORCE the other axis using precomputed metrics
        if self.allow_fallback_axis:
            fallback_axis = "y" if best_axis == "x" else "x"
            try:
                fb = overall_metrics.get(fallback_axis)
                if fb is None:
                    return False
                fb_div = fb["_division_point"]
                fb_subL = fb["_subregion_left"]
                fb_subR = fb["_subregion_right"]
                return self._attempt_partition(region, obstacles, fallback_axis, depth, fb_div, fb_subL, fb_subR, parent_id=node_id)
            except Exception:
                return False

        return False

    # ✅ SIGNATURE CHANGE: add parent_id (additive)
    def _attempt_partition(self, region, obstacles, axis, depth, division_point, subL, subR, parent_id: int) -> bool:
        self.axis_stack.append(axis)

        # --- Numerical stability: clamp cut inside open interval (min+eps, max-eps) ---
        minx, miny, maxx, maxy = region.bounds
        eps = self._EPS
        if axis == "x":
            if division_point is None:
                self.axis_stack.pop()
                return False
            division_point = min(max(division_point, minx + eps), maxx - eps)
            if division_point <= minx + eps or division_point >= maxx - eps:
                self.axis_stack.pop()
                return False
        else:
            if division_point is None:
                self.axis_stack.pop()
                return False
            division_point = min(max(division_point, miny + eps), maxy - eps)
            if division_point <= miny + eps or division_point >= maxy - eps:
                self.axis_stack.pop()
                return False

        (R_left, left_obs) = subL
        (R_right, right_obs) = subR

        left_ok = self._is_subregion_valid(R_left, left_obs)
        right_ok = self._is_subregion_valid(R_right, right_obs)

        if not left_ok and not right_ok:
            self.axis_stack.pop()
            return False

        produced_any = False

        # ---- LEFT child ----
        if left_ok:
            self._addr_bits_stack.append("0")
            produced_any = self._decompose(R_left, left_obs, depth + 1, parent_id=parent_id) or produced_any
            self._addr_bits_stack.pop()

        # ---- RIGHT child ----
        if right_ok:
            self._addr_bits_stack.append("1")
            produced_any = self._decompose(R_right, right_obs, depth + 1, parent_id=parent_id) or produced_any
            self._addr_bits_stack.pop()

        self.axis_stack.pop()
        return produced_any

    def _is_subregion_valid(self, region, obstacles, largest_hole=None):
        if region is None or region.is_empty:
            return False
        if not region.is_valid:
            return False

        minx, miny, maxx, maxy = region.bounds
        if (maxx - minx) < self.min_dimension_threshold or (maxy - miny) < self.min_dimension_threshold:
            return False

        region_area = region.area
        obs_area_sum = sum(ob.area for ob in obstacles) if obstacles else 0.0
        if obs_area_sum >= region_area - 1e-9:
            return False

        coverage_ratio = (obs_area_sum / region_area) if region_area > 1e-12 else 1.0
        if coverage_ratio >= COVERAGE_RATIO_STOP:
            if largest_hole is None:
                largest_hole = self._compute_largest_free_space(region, obstacles)
            if largest_hole < DRONE_THRESHOLD:
                return False

        if self.check_connectivity:
            if not self._has_single_connected_free_space(region, obstacles):
                return False

        return True

    def _compute_largest_free_space(self, region, obstacles):
        try:
            if not obstacles:
                return region.area

            union_obs = unary_union(obstacles)
            if union_obs.is_empty:
                return region.area

            free_space = region.difference(union_obs)
            if free_space.is_empty:
                return 0.0
            if isinstance(free_space, Polygon):
                return free_space.area
            if isinstance(free_space, MultiPolygon):
                return max(poly.area for poly in free_space.geoms)

            max_area = 0.0
            for g in getattr(free_space, "geoms", []):
                if g.geom_type == "Polygon":
                    max_area = max(max_area, g.area)
                elif g.geom_type == "MultiPolygon":
                    max_area = max(max_area, max(poly.area for poly in g.geoms))
            return max_area
        except Exception:
            return 0.0

    def _has_single_connected_free_space(self, region, obstacles):
        return self._compute_largest_free_space(region, obstacles) >= DRONE_THRESHOLD
