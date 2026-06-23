# Spectroscopy Data Analysis Pipeline for Metalens Characterisation

## Overview

This repository contains a Python-based data analysis pipeline developed during my Honours research in Physics. The code was designed to process, analyse, and visualise experimental spectroscopy data collected from a fabricated metalens device, and to compare it against theoretical models of its optical response.

The workflow focuses on extracting spectral features, fitting resonance profiles, and quantifying key optical parameters such as peak wavelength and full width at half maximum (FWHM).

## Research Context

During my Honours project, I worked with experimentally acquired spectroscopy datasets from a fabricated metalens device. The goal was to:

- Compare experimentally measured optical spectra against theoretical predictions
- Characterise resonance behaviour in nanostructured optical systems
- Quantify spectral features using robust curve fitting techniques
- Generate publication-quality figures for thesis and research outputs

This involved developing a reproducible analysis pipeline capable of handling raw instrument output, extracting meaningful physical signals, and fitting both Gaussian and Fano resonance models.

## Key Features

### Data Handling
- Parses raw spectroscopy `.txt` files with embedded metadata
- Extracts wavelength and intensity matrices
- Supports region-of-interest (ROI) selection

### Signal Processing
- ROI-based spectrum extraction (sum or average)
- Global or local intensity normalisation
- Noise-aware numerical processing

### Curve Fitting
- Gaussian resonance fitting
- Fano resonance fitting (asymmetric line-shapes)
- Robust parameter estimation using `scipy.optimize.curve_fit`

### Feature Extraction
- Peak wavelength identification
- FWHM calculation (analytical and numerical methods)
- Goodness-of-fit estimation (R² for Fano model)

### Visualisation
- Publication-quality figures (300 DPI)
- Spatial-spectral intensity maps
- Overlaid experimental vs fitted spectra
- Thesis-ready formatting using consistent styling

### Output Options
- Save processed spectra to CSV
- Export figures as PNG and PDF
- Structured result summaries for downstream analysis

## Example Workflow

1. Load spectroscopy files from directory
2. Parse metadata (metalens ID, zoom, exposure time, etc.)
3. Extract spectral region of interest
4. Normalise intensity data
5. Fit Gaussian and Fano resonance models
6. Extract physical parameters (peak wavelength, FWHM)
7. Generate and save figures and processed datasets

Outputs
- Spectral intensity heatmaps
- Normalised transmission spectra
- Fitted Gaussian / Fano resonance curves
- Extracted numerical parameters (CSV)
- Publication-ready figures (PNG + PDF)

## Technical Highlights
- Vectorised NumPy operations for efficient spectral processing
- Robust curve fitting with bounded parameter optimisation
- Dual-model fitting (Gaussian vs Fano resonance comparison)
- Automated figure generation for thesis and publication use
- Modular, class-based architecture for scalability

## Skills
- Scientific Python programming
- Experimental data analysis
- Curve fitting and optimisation
- Signal processing and feature extraction
- Research-grade data visualisation
- Reproducible computational workflows
- Potential Extensions

## Future adaptations of this pipeline could include:
- Integration with previous image analysis code to pre-define ROI and other parameters loaded on to file
