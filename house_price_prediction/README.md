# 🏠 House Price Prediction (AI & ML Virtual Internship Project)

An end-to-end machine learning project that predicts house prices (in INR lakhs)
from property attributes: preprocessing → model training → evaluation → web interface → tests.

## Project structure
```
house_price_prediction/
├── app.py                  # Streamlit prediction interface
├── requirements.txt
├── data/housing.csv        # dataset (synthetic, Bengaluru-style; swappable)
├── src/
│   ├── config.py           # paths, feature lists, constants
│   ├── generate_data.py    # creates the dataset (with deliberate messy data)
│   ├── preprocess.py       # cleaning + imputation/scaling/encoding pipeline
│   ├── train.py            # model comparison, tuning, saving
│   ├── evaluate.py         # metrics + diagnostic plots
│   └── predict.py          # input validation + inference
├── tests/test_pipeline.py  # 14 pytest tests
├── models/                 # saved model (joblib)
├── reports/                # metrics.json + plots
└── docs/REPORT.md          # full project report
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.generate_data     # 1. create data/housing.csv
python -m src.train             # 2. train, evaluate, save model + reports
python -m pytest -q             # 3. run tests
streamlit run app.py            # 4. launch the prediction UI
```

## Workflow
| Stage | What happens |
|---|---|
| Data cleaning | Remove duplicates, invalid/negative prices, and outliers (IQR on price per sq.ft) |
| Preprocessing | Median imputation + scaling (numeric); mode imputation + one-hot (categorical), inside a `Pipeline` so there is no train/test leakage |
| Training | 4 models compared with 5-fold CV; best one tuned with `GridSearchCV` |
| Evaluation | MAE, RMSE, R², MAPE on a held-out 20% test set; plots saved to `reports/` |
| Interface | Streamlit form with input validation and a price + price/sq.ft output |
| Testing | Cleaning, missing values, unseen categories, model quality, sanity checks, validation |

## Results (hold-out test set)
| Metric | Value |
|---|---|
| R² | 0.957 |
| MAE | ₹7.8 lakhs |
| RMSE | ₹11.2 lakhs |
| MAPE | 7.0 % |

Best model: **Gradient Boosting** (CV RMSE 14.9 vs 18.0 for linear models).

## Using real data
Replace `data/housing.csv` with a real dataset containing the same columns
(or edit `src/config.py` to match your columns), then re-run `python -m src.train`.
