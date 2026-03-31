# Synthetic Data Generation

This folder contains scripts for generating synthetic obstacle datasets used in OA-BAR and RBA experiments.

## Main script

- `research_data_generator.py`  
  Generates synthetic obstacle fields with configurable coverage, seeds, obstacle-size controls, optional clustered placement, connectivity checks, and JSON metadata.

## Optional companion script

- `non_uniform_generator.py`  
  Used for clustered or non-uniform obstacle placement configurations when needed.

## Output

Generated datasets are stored as JSON files containing:
- region geometry
- obstacle geometries
- metadata such as target coverage, achieved coverage, seed, and generator parameters