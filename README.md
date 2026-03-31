# Obstacle-Aware-Spatial-Partitioning-and-Allocation

Research code for obstacle-aware hierarchical spatial partitioning and hierarchy-constrained apportionment under worst-case response bounds.

This repository contains two main components:

- **OA-BAR**: an obstacle-aware balanced hierarchical partitioning framework that constructs axis-aligned spatial decompositions by balancing an upper bound on the longest shortest path (ULSP) while maintaining geometric regularity.
- **RBA**: a hierarchy-constrained apportionment framework that supports arbitrary fleet sizes by distributing additional splits across the OA-BAR hierarchy while preserving structural balance.

Together, these components support region assignment for mobile surveillance and response settings in obstacle-rich environments, with emphasis on worst-case response behavior, hierarchical fairness, and practical spatial decomposition.

## Repository Structure

```text
src/
  oabar/
  rba/
  baselines/
  common/
  utils/
experiments/
  oabar/
  rba/
data/
figures/
scripts/