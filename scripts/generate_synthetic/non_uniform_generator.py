import os
import random
import math
import json
import logging
import numpy as np
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

logging.basicConfig(level=logging.INFO)


class ResearchDataGenerator:
    def __init__(self, region_size=(100, 100), obstacle_percentage=50,
                 max_obstacle_size=None, size_variation=0.2, min_obstacle_size=0.5,
                 seed=None, clusters=None):
        self.region_size = region_size
        self.obstacle_percentage = obstacle_percentage
        self.max_obstacle_size = max_obstacle_size
        self.size_variation = size_variation
        self.min_obstacle_size = min_obstacle_size
        self.clusters = clusters or []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_region(self):
        width, height = self.region_size
        return Polygon([(0, 0), (width, 0), (width, height), (0, height)])

    def generate_convex_polygon(self, center, size, sides):
        angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(sides))
        points = [(center[0] + size * math.cos(angle),
                   center[1] + size * math.sin(angle)) for angle in angles]
        return Polygon(points).convex_hull

    def sample_center(self):
        width, height = self.region_size
        if not self.clusters:
            return (random.uniform(0, width), random.uniform(0, height))

        weights = [cluster['weight'] for cluster in self.clusters]
        probabilities = [w / sum(weights) for w in weights]
        chosen_cluster = np.random.choice(self.clusters, p=probabilities)
        cx, cy = chosen_cluster['center']
        std_dev = chosen_cluster['std_dev']
        x = np.random.normal(cx, std_dev)
        y = np.random.normal(cy, std_dev)
        x = min(max(x, 0), width)
        y = min(max(y, 0), height)
        return (x, y)

    def generate_obstacles(self, region):
        # Target obstacle union area
        target_area = (self.obstacle_percentage / 100) * region.area
        obstacles = []
        current_union = 0.0

        no_progress_limit = 5000  # number of failed attempts allowed
        no_progress_counter = 0

        # Adaptive phase: use variable obstacle sizes based on remaining area.
        while current_union < target_area and no_progress_counter < no_progress_limit:
            remaining_area = target_area - current_union
            scaling_factor = 10  # adjust based on region and target
            base_size = math.sqrt(remaining_area / scaling_factor)

            # Apply variation so obstacles are not all the same size.
            size = base_size * (1 + self.size_variation * random.uniform(-1, 1))
            # Ensure size is at least the minimum.
            size = max(size, self.min_obstacle_size)

            if self.max_obstacle_size:
                max_size_limit = math.sqrt(self.max_obstacle_size / math.pi)
                size = min(size, max_size_limit)

            sides = random.choice([3, 4, 5, 6])
            center = self.sample_center()
            obstacle = self.generate_convex_polygon(center, size, sides)

            # Check that the new obstacle is completely within the region.
            if region.contains(obstacle) and obstacle.is_valid:
                # Also check that it does not intersect any already accepted obstacle.
                if all(not obstacle.intersects(existing) for existing in obstacles):
                    obstacles.append(obstacle)
                    union_poly = unary_union(obstacles)
                    new_union = union_poly.area
                    # Only count if we actually increased the union area.
                    if new_union > current_union:
                        current_union = new_union
                        no_progress_counter = 0
                        continue

            no_progress_counter += 1

        # If we didn't reach the target, try one more pass with fixed minimal size obstacles.
        if current_union < target_area:
            logging.info("Switching to fixed small obstacles to fill gaps.")
            fill_attempts = 0
            max_fill_attempts = 10000
            fixed_size = self.min_obstacle_size
            while current_union < target_area and fill_attempts < max_fill_attempts:
                center = self.sample_center()
                sides = random.choice([3, 4, 5, 6])
                obstacle = self.generate_convex_polygon(center, fixed_size, sides)
                if region.contains(obstacle) and obstacle.is_valid:
                    if all(not obstacle.intersects(existing) for existing in obstacles):
                        obstacles.append(obstacle)
                        union_poly = unary_union(obstacles)
                        current_union = union_poly.area
                fill_attempts += 1

            if current_union < target_area:
                logging.warning(
                    f"Even after fill phase, target area not reached. Target: {target_area:.2f}, Achieved: {current_union:.2f}")

        return obstacles

    def ensure_connectivity(self, region, obstacles):
        valid_obstacles = []
        for obstacle in obstacles:
            temp_region = region.difference(obstacle)
            if temp_region.geom_type in ["Polygon", "MultiPolygon"]:
                valid_obstacles.append(obstacle)
            else:
                logging.warning("Obstacle discarded to maintain connectivity.")
        return valid_obstacles

    def generate_and_store(self, file_path):
        region = self.generate_region()
        obstacles = self.generate_obstacles(region)
        obstacles = self.ensure_connectivity(region, obstacles)

        union_poly = unary_union(obstacles)
        total_obstacle_area = union_poly.area
        data = {
            "region": mapping(region),
            "obstacles": [mapping(obstacle) for obstacle in obstacles],
            "total_obstacle_area": total_obstacle_area,
            "target_obstacle_area": (self.obstacle_percentage / 100) * region.area
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        logging.info(f"Synthetic data saved to {file_path}.")
        logging.info(
            f"Target Area: {data['target_obstacle_area']:.2f}, Achieved Obstacle Union Area: {total_obstacle_area:.2f}, Count: {len(obstacles)}")

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        region = shape(data["region"])
        obstacles = [shape(obstacle) for obstacle in data["obstacles"]]
        return region, obstacles, data.get("total_obstacle_area", 0), data.get("target_obstacle_area", 0)

    def visualize_obstacles(self, region, obstacles, title="5% Obstacle coverage"):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_polygon(region, ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Region')
        # label = 'Obstacle' if i == 0 else None
        for i, obstacle in enumerate(obstacles):
            label = 'Obstacle' if i == 0 else None
            plot_polygon(obstacle, ax=ax, facecolor='red', edgecolor='darkred', alpha=0.6, label=label)
        # ax.set_title(title, fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Testing the modified generator for a 50% obstacle coverage
if __name__ == "__main__":
    region_size = (100, 100)
    max_obstacle_size = 50  # adjust as needed
    size_variation = 0.2  # adaptive phase variation
    min_obstacle_size = 0.5  # fixed small obstacle size for filling gaps
    seed = 73

    clusters = [

    ]

    obstacle_percentage = 5
    logging.info(f"Generating data for {obstacle_percentage}% obstacle coverage.")
    generator = ResearchDataGenerator(
        region_size=region_size,
        obstacle_percentage=obstacle_percentage,
        max_obstacle_size=max_obstacle_size,
        size_variation=size_variation,
        min_obstacle_size=min_obstacle_size,
        seed=seed,
        clusters=clusters
    )

    file_path = f"synthetic_data_generation/modified/RBA/5_percent_obstacles.json"
    generator.generate_and_store(file_path)

    region, obstacles, total_area, target_area = ResearchDataGenerator.load_from_file(file_path)
    achieved_coverage = (total_area / region.area) * 100 if region.area else 0
    logging.info(f"Target: {obstacle_percentage}% region coverage. "
                 f"Generated {len(obstacles)} obstacles, "
                 f"Union Obstacle Area: {total_area:.2f}, Achieved Coverage: {achieved_coverage:.2f}%.")
    generator.visualize_obstacles(region, obstacles)
