# Deep Learning for PJM Day-Ahead Electricity Price Forecasting

Forecasting hourly PJM-RTO day-ahead electricity prices using leakage-safe features, classical baselines, and an LSTM sequence model.

## Overview

This project studies whether historical price data, lagged system variables, and calendar features available before the day-ahead market clears can improve hourly electricity price forecasts.

We compare three model families:

- Persistence baselines
- Random Forest
- LSTM sequence model

The final LSTM achieves a test MAE of 7.69 and RMSE of 13.86 on the held-out test set. The Random Forest is a strong benchmark and slightly improves RMSE, so the main takeaway is that leakage-safe machine learning clearly beats naive persistence.

## Repository Layout

- `data/` raw PJM source files used to build the feature table
- `data_preprocess.ipynb` preprocessing pipeline and feature engineering
- `baseline_modeling.ipynb` baseline experiments and classical models
- `final_model.ipynb` main modeling notebook
- `lstm_outputs/` saved metrics, predictions, feature importance, and model artifacts
- `plots/` figures used in the report
- `report/` final LaTeX write-up
- `initial_data.csv` merged hourly dataset

## Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If you plan to train or rerun the neural network experiments, make sure you have a PyTorch build that matches your system. For CUDA-enabled installs, follow the selector on the [official PyTorch site](https://pytorch.org/get-started/locally/).

## Suggested Workflow

1. Start with `data_preprocess.ipynb` to rebuild the hourly feature table.
2. Run `baseline_modeling.ipynb` to reproduce the baseline comparisons.
3. Open `final_model.ipynb` for the final LSTM experiments and saved outputs.

## Key Results

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Persistence (48h) | 14.72 | 23.71 |
| Random Forest | 8.04 | 13.26 |
| Final LSTM | 7.69 | 13.86 |

## Notes

- The feature set is designed to be leakage-safe for the PJM day-ahead cutoff.
- Outputs in `lstm_outputs/` include `model_metrics.csv`, `test_predictions.csv`, and feature-importance files.
- The final report and figures are stored in `report/` and `plots/`.

## Clone

```bash
git clone https://github.com/anthony240624/CS109B-Final-Project
cd CS109B-Final-Project
```