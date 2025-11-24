# Stock Returns & Volatility Prediction

Python notebooks to engineer features from Yahoo Finance, explore signals, train classifiers for next-day direction / volatility spikes, and explain and backtest models.

## Repository layout
- `GP Data Creation.ipynb` – downloads prices (2015–2025) for 14 tickers, builds technical + VIX features, labels targets, saves `data/merged_features.csv`.
- `EDA_Insights.ipynb` – sanity-checks the dataset with distributions, correlations, and event-window plots.
- `Modeling.ipynb` – trains Logistic Regression (plus calibrated variant), Random Forest, HistGradientBoosting, and XGBoost on next-day direction; includes coin-flip / repeat-last baselines and confusion matrices.
- `Explainability Backtesting.ipynb` – SHAP importance for XGBoost, walk-forward check for DAL, long/short equity curve vs buy-and-hold, and backtest summary stats.
- `data/merged_features.csv` – sample feature set (regenerate if you run data creation).
- `requirements.txt` – project dependencies.

## Setup
1) Python 3.10+ recommended.  
2) Install deps: `pip install -r requirements.txt`  
   - If you’re offline, skip data creation and use `data/merged_features.csv`.  
3) Run notebooks in order: GP data creation → EDA → Modeling → Explainability/backtesting.

## Reproduce the data
- Open `GP Data Creation.ipynb`, run cells.  
- Output: `data/merged_features.csv` (created automatically).  
- Labels: `Direction` (next-day up/down) and `VolSpike` (rolling vol > 80th pct).

## Modeling notes
- Splits are per-ticker chronological (80/20) to avoid look-ahead across stocks.  
- Features standardized for linear models; tree models use raw features.  
- Baselines included: coin flip, repeat-last-direction, majority class.  
- Models compared: Logistic Regression with a calibrated version, Random Forest, HistGradientBoosting, XGBoost.  
- Evaluate with accuracy, F1, ROC AUC; baselines help contextualize lift.