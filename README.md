# Magnetospheric B-Duct Detection

This project is a prototype pipeline for AI-enabled detection and characterization of magnetospheric ducting structures from MMS observations.

## Example duct signatures

**Series of Detected Events in (A) Flank, and (A') Magnetotail of Magnetosphere**
<p align="center">
  <img src="assets/Fig1-2.png" alt="Detected duct events frSom MMS observations" width="95%">
</p>

<p align="center"><sub>Detected duct candidates used as examples for the AI detection workflow.</sub></p>


<p align="center">
  <img src="assets/Fig1-2.png" alt="ELF wave packets localized in the $B-$ducts observed by MMS-1 satellite in phase-1 on March 06, 2016, from 23:25:44 to 23:27:59 UT, and in phase-2 from 1:44:20 to 1:47:40 UT on July 21, 2017. Panels (A) and (A') show the power spectral density (PSD) of $E_x$ component of the electric field in the GSE coordinate system of the satellite (shown with a color pallet) and background magnetic field (shown with the white line). Panels (B) and (B') show the electron density measured by the MMS1 satellite. Panel (C) and (C') show the trajectory and the location of MMS satellites in the GSE X-Y plane on March 6, 2016 (19:00 UT), and July 21, 2017 (2:00 UT), respectively." width="95%">
</p>

**Observed (A) High-magnetic, (A') Low-Magnetic, and (A'') Magnetic-Shelf Events**

<p align="center">
  <img src="assets/B_ducts.png" alt="Examples of magnetic-field duct signatures in MMS observations" width="95%">
</p>



## Workflow

- Identify observation intervals that contain possible magnetospheric duct signatures.
- Extract event-centered magnetic-field and plasma-density structures.
- Use the extracted samples to train and test AI models for individual duct detection.
- Compare detected events with observed duct signatures across different regions of the magnetosphere.

## Status

Research prototype under development. The project currently focuses on organizing duct-event samples, preparing detected-event examples, and building an AI workflow for identifying individual magnetospheric ducting structures from observational data.

## Repository status
- **Current maturity:** structured research prototype suitable for a TRL 4–5 starting point
- **Legacy development notebooks:** preserved under `archive/legacy_notebooks/`
- **Public-facing workflow notebooks:** kept in `notebooks/` and aligned to the packaged code in `src/`

## Pipeline stages
1. **Data ingestion** – download MMS products, interpolate to a common timeline, and export interval products.
2. **Event dataset construction** – build event-centered NetCDF samples and quicklook plots.
3. **Feature preparation** – convert event samples into ML-ready tabular features and spectrogram tensors.
4. **Model development** – train baseline and advanced models from prepared arrays.

## Quick start
```bash
pip install -e .
```

## Main workflow notebooks
- `notebooks/01_mms_data_ingest_interpolation.ipynb`
- `notebooks/02_event_dataset_and_quicklooks.ipynb`
- `notebooks/03_feature_preparation.ipynb`
- `notebooks/04_model_random_forest.ipynb`

## Labels
Model training requires a label table. A template is provided at `labels/example_event_labels.csv` with columns:
- `file`
- `label`

## Notes
- Large data products are intentionally excluded from version control.
- Historical notebooks are retained for development traceability but should not be treated as the current workflow.
