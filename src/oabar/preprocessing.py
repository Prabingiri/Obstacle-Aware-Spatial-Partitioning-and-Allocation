from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, transform
from shapely.validation import explain_validity
from shapely.errors import TopologicalError
import logging

logging.basicConfig(level=logging.INFO)


class RegionWithObstacles:
    """
    Preprocess and validate a region with obstacles.

    Fixes invalid geometries, clips obstacles to region, and merges overlaps.
    """

    def __init__(self, region_geom, obstacle_coords_list):
        self.region = self._validate_and_fix_region(region_geom)
        self.obstacles = self._create_and_clip_obstacles(obstacle_coords_list)
        self.merged_obstacles = self._merge_obstacles(self.obstacles)

    @staticmethod
    def drop_z_coordinates(geometry):
        if geometry is None:
            return geometry
        if getattr(geometry, "has_z", False):
            geometry = transform(lambda x, y, z=None: (x, y), geometry)
        return geometry

    def _validate_and_fix_region(self, region_geom):
        region_geom = self.drop_z_coordinates(region_geom)

        if not isinstance(region_geom, (Polygon, MultiPolygon)):
            raise ValueError("The region must be a Polygon or MultiPolygon.")

        if not region_geom.is_valid:
            logging.warning("Region geometry is invalid. Attempting to fix with buffer(0).")
            region_geom = region_geom.buffer(0)

        if not region_geom.is_valid:
            raise ValueError(f"Region geometry is invalid: {explain_validity(region_geom)}")

        return region_geom

    # ---- small helper: keep only polygonal parts & flatten to list[Polygon] ----
    def _extract_polygons(self, geom):
        """
        Convert an arbitrary shapely geometry to a list of Polygon parts.
        Keeps polygonal components only; drops lines/points.
        """
        if geom is None or geom.is_empty:
            return []

        if isinstance(geom, Polygon):
            return [geom]

        if isinstance(geom, MultiPolygon):
            return list(geom.geoms)

        if isinstance(geom, GeometryCollection):
            polys = []
            for g in geom.geoms:
                polys.extend(self._extract_polygons(g))
            return polys

        # other geometry types (LineString, Point, etc.)
        return []

    def _create_and_clip_obstacles(self, obstacle_coords_list):
        obstacles = []

        for i, coords in enumerate(obstacle_coords_list, start=1):
            obstacle = self._create_and_validate_polygon(coords, f"Obstacle {i}")
            if obstacle is None or obstacle.is_empty:
                logging.warning(f"Obstacle {i} is invalid/empty and was discarded.")
                continue

            obstacle = self.drop_z_coordinates(obstacle)

            try:
                clipped = obstacle.intersection(self.region)
            except TopologicalError as e:
                logging.warning(f"Obstacle {i} intersection TopologicalError: {e}. Trying buffer(0) fix.")
                try:
                    clipped = obstacle.buffer(0).intersection(self.region)
                except Exception as e2:
                    logging.warning(f"Obstacle {i} still failed intersection after fix: {e2}. Discarding.")
                    continue

            clipped_polys = self._extract_polygons(clipped)
            if not clipped_polys:
                logging.warning(f"Obstacle {i} lies outside the region (or non-polygon result) and was discarded.")
                continue

            # ensure each clipped polygon is valid (minimal, safe)
            for p in clipped_polys:
                if not p.is_valid:
                    p2 = p.buffer(0)
                    if p2.is_valid and (not p2.is_empty):
                        obstacles.extend(self._extract_polygons(p2))
                    else:
                        logging.warning(f"Obstacle {i} produced invalid clipped polygon and was discarded.")
                else:
                    obstacles.append(p)

        return obstacles

    def _merge_obstacles(self, obstacles):
        if not obstacles:
            logging.warning("No obstacles to merge.")
            return []

        try:
            merged = unary_union(obstacles)
            merged_polys = self._extract_polygons(merged)

            if not merged_polys:
                logging.warning("Merged obstacles result is empty (or non-polygon).")
                return []

            # final validity pass (cheap)
            out = []
            for p in merged_polys:
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and (not p.is_empty):
                    out.append(p)
            return out

        except TopologicalError as e:
            logging.error(f"Error during obstacle merging: {e}")
            # minimal fallback: return original list (already clipped + polygonal)
            return obstacles

    def _create_and_validate_polygon(self, coords, name):
        """
        Minimal change: attempt to FIX invalid polygon instead of discarding immediately.
        """
        polygon = Polygon(coords)
        polygon = self.drop_z_coordinates(polygon)

        if polygon.is_empty:
            logging.warning(f"{name} is empty and will be discarded.")
            return None

        if not polygon.is_valid:
            validity = explain_validity(polygon)
            logging.warning(f"{name} is invalid ({validity}). Trying buffer(0) fix.")
            polygon2 = polygon.buffer(0)
            if polygon2.is_empty or (not polygon2.is_valid):
                logging.warning(f"{name} could not be fixed and will be discarded ({explain_validity(polygon2)}).")
                return None
            polygon = polygon2

        return polygon

    def get_simplified_obstacles(self):
        return self.merged_obstacles

    def check_region_connectivity(self):
        remaining_region = self.region
        try:
            merged_obstacles = unary_union(self.merged_obstacles) if self.merged_obstacles else None
            if merged_obstacles and (not merged_obstacles.is_empty):
                remaining_region = remaining_region.difference(merged_obstacles)
        except TopologicalError as e:
            logging.error(f"Error during connectivity check: {e}")
            return False

        # connected free space => single Polygon (simple test, consistent with your previous logic)
        return remaining_region.geom_type == "Polygon"

    def create_region(self, region_geom):
        region_geom = self.drop_z_coordinates(region_geom)
        if not isinstance(region_geom, (Polygon, MultiPolygon)):
            raise ValueError("The region must be a Polygon or MultiPolygon.")

        if not region_geom.is_valid:
            logging.warning("Region geometry is invalid. Attempting to make it valid.")
            region_geom = region_geom.buffer(0)

        if not region_geom.is_valid:
            raise ValueError(f"The region geometry is invalid: {explain_validity(region_geom)}")

        return region_geom

    def visualize(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the region
        if isinstance(self.region, Polygon):
            x, y = self.region.exterior.xy
            ax.plot(x, y, color='black', label='Region Boundary')
        elif isinstance(self.region, MultiPolygon):
            for j, p in enumerate(self.region.geoms):
                x, y = p.exterior.xy
                ax.plot(x, y, color='black', label='Region Boundary' if j == 0 else None)

        # Plot raw obstacles (already flattened to polygons now)
        for j, obstacle in enumerate(self.obstacles):
            x, y = obstacle.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5, label='Obstacle' if j == 0 else None)

        # Plot merged obstacles
        for j, geom in enumerate(self.merged_obstacles):
            x, y = geom.exterior.xy
            ax.plot(x, y, color='blue', linestyle='--', label='Merged Obstacle' if j == 0 else None)

        ax.set_title("Region with Obstacles")
        ax.legend()
        plt.show()
