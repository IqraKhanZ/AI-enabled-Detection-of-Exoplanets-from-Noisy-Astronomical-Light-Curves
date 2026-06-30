# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

An end-to-end AI-driven pipeline that downloads, preprocesses, and classifies TESS high-cadence astronomical light curves to detect planetary transits, and extracts candidate physical parameters (orbital period, transit depth, and transit duration) using Markov Chain Monte Carlo (MCMC).

🔗 **Live Dashboard**: [https://ai-enabled-detection-of-exoplanets-from.onrender.com/](https://ai-enabled-detection-of-exoplanets-from.onrender.com/)

## 1. Overview
Exoplanet detection through transit photometry requires identifying extremely small brightness variations in stars (often tens of parts-per-million). In crowded fields, stellar blending, instrumental noise, and intrinsic stellar variability obscure transit signals.

This pipeline:
1. **Acquires** TESS Sector short-cadence data & Gaia DR3 cross-matches.
2. **Conditions** signals using multi-resolution wavelet detrending and Gaussian Processes.
3. **Classifies** candidates into `PLANET`, `ECLIPSING_BINARY`, `BLEND`, or `NOISE` using a hybrid 1D-Transformer + 2D-CNN neural network.
4. **Estimates** transit parameters and confidence intervals using Markov Chain Monte Carlo (MCMC) with a physical transit model (`batman`) + GP noise model (`celerite2`).
5. **Visualizes** candidates, posteriors, and field diagnostics via an interactive Dash dashboard.

---

## 2. Pipeline Architecture

```
                 TESS Light Curves (.fits) & Target Pixel Files (TPF)
                                      │
                                      ▼
                      [ Systematics & Wavelet Detrending ]
                                      │
                                      ▼
                     [ Gaussian Process Detrending ]
                                      │
                                      ▼
                        [ BLS Search & Phase-Folding ]
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
      [ 2D River Plot View ]                      [ 1D Global & Local View ]
                │                                           │
                ▼                                           ▼
          { 2D-CNN Branch }                       { 1D-Transformer Branch }
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      ▼
                           [ Cross-Attention Fusion ]
                                      │
                                      ▼
                          [ Classifier Predictions ]
                   (Planet / Eclipse / Blend / Noise)
                                      │
                                      ▼ (If Planet)
                      [ Transit Fitter & MCMC Sampling ]
                        (batman model + celerite2 GP)
                                      │
                                      ▼
                       [ Plotly Dash Dashboard & Reports ]
```

---

## 3. Installation

We recommend using a conda environment to install dependencies:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate exoplanet-pipeline

# Or install dependencies via pip
pip install -r requirements.txt

# Install the pipeline package in development mode
pip install -e .
```

---

## 4. Quick Start

Run these commands sequentially to run the pipeline from end to end:

```bash
# 1. Download data and labels for TESS Sector 1 (restricted to 100 targets for speed)
python src/acquisition/download_lightcurves.py --sector 1 --max-targets 100

# 2. Run quality control checks to flag corrupted or flat files
python src/preparation/quality_control.py

# 3. Stratify and split the labeled catalog into train/validation/test sets
python src/preparation/train_val_test_split.py

# 4. Run the full pipeline (preprocessing, classifier training, MCMC fitting, and outputs)
python src/pipeline/run_pipeline.py --sector 1 --max-targets 100

# 5. Start the web dashboard to view results
python src/visualization/dashboard.py
```

---

## 5. Configuration (`configs/config.yaml`)

The pipeline parameters are centralized in `configs/config.yaml`. Key sections include:
- `acquisition.sector`: TESS sector to target.
- `quality_control`: Thresholds for NaN fractions, duration, and crowding.
- `conditioning`: Wavelet levels and BLS periodogram parameters.
- `model`: Neural network architecture sizes for the Transformer and CNN branches.
- `mcmc`: Walker count and step limits for MCMC parameter estimation.

---

## 6. Output Format

Results are aggregated into `outputs/pipeline_results.csv`. Schema columns include:
* `tic_id`: TESS Input Catalog Identifier.
* `predicted_label_name`: Predicted class (`PLANET`, `ECLIPSING_BINARY`, `BLEND`, `NOISE`).
* `pipeline_confidence`: Integrates classifier probability, SNR, and FAP.
* `period_days`: Fitted orbital period.
* `depth_ppm`: Fitted transit depth in parts-per-million.
* `duration_hrs`: Fitted transit duration.
* `snr`: Signal-to-Noise Ratio of the transit dip.
* `contamination_ratio`: Contamination from nearby stars inside target aperture.

---

## 7. Interactive Dashboard

Start the Dash web application:
```bash
python src/visualization/dashboard.py
```
Open [http://localhost:8050](http://localhost:8050) in your browser. The dashboard lets you:
* Search for any processed TIC ID.
* Toggle between raw, detrended, and folded light curves.
* Review classification class probabilities and gauges.
* View physical parameters and uncertainties.
* Inspect SNR and target-aperture stellar contamination.

---

## 8. Running Tests

Run the test suite to verify pipeline integrity:
```bash
pytest tests/ -v
```

---

## 9. Directory Structure

```
├── configs/
│   └── config.yaml             # Central configuration
├── data/
│   ├── raw/                    # Downloaded light curves and TPFs
│   ├── interim/                # Preprocessing cache files
│   └── processed/              # QC flags and training splits
├── src/
│   ├── acquisition/            # Download and metadata scripts
│   ├── preparation/            # QC and partitioning
│   ├── conditioning/           # Signal detrending and folding
│   ├── features/               # Feature engineering
│   ├── models/                 # Transformer/CNN architectures & MCMC
│   ├── scoring/                # SNR, FAP, and confidence scores
│   ├── pipeline/               # Orchestrator
│   └── visualization/          # Standalone plots and Dash dashboard
├── tests/                      # Pytest verification suite
├── setup.py                    # Package installer
└── README.md                   # This file
```

---

## 10. License
This project is licensed under the MIT License.
