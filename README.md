# magnetospheric $B$-duct detection

AI-enabled detection and characterization of magnetospheric ducting structures from space physics observations.

## Current status
This repository contains the prototype pipeline for:
- MMS data ingestion and interpolation
- preprocessing of duct-relevant intervals
- downstream machine-learning workflows for detection and characterization

## Repository structure
- `notebooks/` exploratory and workflow notebooks
- `src/` Python modules
- `data/` raw and processed data products
- `results/` figures and model outputs

## First implemented component
The current prototype includes an MMS data-ingestion notebook for downloading, interpolating, and exporting satellite observations for selected intervals.
