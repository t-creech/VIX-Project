# VIX Project

This project, undertaken for the FIM500 course, explores forecasting the CBOE Volatility Index (VIX). The VIX measures the market's expectation of 30-day volatility derived from S&P 500 option prices and is widely used as a gauge of market sentiment. Accurate forecasting of the VIX can help investors manage risk, design option strategies, and assess economic uncertainty.

## Overview

The repository contains a suite of Jupyter notebooks investigating various modelling approaches for the VIX time series, including:
- **Exploratory data analysis** to visualize VIX dynamics and relationships with macro indicators.
- **Classical time‑series models** such as ARMA and SARIMA.
- **Machine learning models** like random forests and gradient boosting.
- **Deep learning methods** such as Long Short-Term Memory (LSTM) networks.

Each notebook documents data preprocessing, model training, evaluation metrics and insights. The aim is to compare the performance of these approaches on the same dataset.

## Repository structure

- `VIX_EDA.ipynb` – exploratory analysis.
- `VIX_ARMA.ipynb` – ARMA modeling.
- `VIX_SARIMA.ipynb` – seasonal ARIMA modeling.
- `VIX_RF.ipynb` – random forest regression for VIX forecasting.
- `VIX_LSTM.ipynb` – LSTM neural network for sequence prediction.
- `data/` – raw and processed data (if included).
- `environment.yml` / `requirements.txt` – Python dependencies.

## Objectives

- Compare forecasting performance across statistical, machine learning and deep learning approaches.
- Evaluate models using metrics such as mean absolute error (MAE) and root mean squared error (RMSE).
- Discuss practical implications of VIX forecasting for risk management and option strategies.

Feel free to use these notebooks as a starting point for your own volatility research and adapt the code to fit new datasets or modelling ideas.
