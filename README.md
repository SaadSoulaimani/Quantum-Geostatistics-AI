# Quantum Geostatistics and Artificial Intelligence

Code for the manuscript **"Quantum Geostatistics and Artificial Intelligence:
A New Approach for Spatial Data Analysis"**.

## What it does

Spatial coordinates are encoded into a small qubit register with a Fourier data re-uploading
feature map (simulated with QuTiP); the fidelity between encoded states is a quantum kernel that
is used inside kriging / Gaussian-process interpolation, alone or blended with a radial-basis kernel.
The method is benchmarked with nested cross-validation, repeated realisations, paired tests and
uncertainty calibration against ordinary kriging (data-driven variogram selection), likelihood-fitted
Gaussian processes (RBF, Matern 3/2, anisotropic) and a spatial random forest, on:

* a synthetic Gaussian random field built on the exact grid of the real survey, and
* the residual gravity and magnetic grids of the Hajjar ore body, Morocco (Soulaimani et al., 2020).

## Files

| File | Description |
|------|-------------|
| `qgeo.py` | Library: real-data loader, synthetic fields, sampling designs, variogram models and selection, kriging, GP, quantum feature map (QuTiP reference + verified vectorised encoder), random forest, uncertainty scores, nested CV, ablation, diagnostics. |
| `Quantum_Geostatistics_AI.ipynb` | Notebook regenerating every table and figure of the paper. |
| `test_example.py` | Quick test (about one minute). |

## Install and run

```bash
pip install -r requirements.txt
python test_example.py                       # quick test, exits 0 on success
jupyter notebook Quantum_Geostatistics_AI.ipynb
```

The residual gravity and magnetic data of the Hajjar ore body used in this study are available from the corresponding author on request. The synthetic fields are generated within the companion notebook of the public repository (https://github.com/SaadSoulaimani/Quantum-Geostatistics-AI) from fixed random seeds and require no separate distribution.

## Reproducibility

All randomness is seeded. The vectorised quantum encoder is checked against the QuTiP
reference inside the notebook (fidelity difference below 1e-15).

## Licence and citation

MIT licence.
