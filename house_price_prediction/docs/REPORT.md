# House Price Prediction – Project Report

## 1. Objective
Build a regression model that estimates a house's price from its attributes and
expose it through a simple web interface.

## 2. Dataset
2,015 raw rows, 10 columns. Target `price_lakhs`. Features: `area_sqft`, `bedrooms`,
`bathrooms`, `age_years`, `parking`, `floor`, `distance_to_metro_km` (numeric) and
`location`, `furnishing` (categorical).

The data is **synthetic**, generated from realistic pricing rules (location rate per
sq.ft, depreciation with age, furnishing premium, metro proximity, market noise) because
no licensed dataset was bundled. Real-world defects were injected on purpose: ~2–3 %
missing values, 15 duplicate rows, 5 negative prices and 8 price outliers.
Because the data is simulated, the high R² reflects how clean the generating rules are;
expect lower scores on real market data.

## 3. Data preprocessing
1. **Cleaning** (before the split): 2,015 → 1,985 rows after dropping duplicates,
   invalid prices and price-per-sq.ft outliers (1.5×IQR).
2. **Pipeline** (fit on training data only): median imputation + `StandardScaler` for numeric
   columns; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical.
3. 80/20 train/test split, `random_state=42`.

## 4. Model training
| Model | 5-fold CV RMSE (lakhs) |
|---|---|
| Linear Regression | 18.01 ± 4.64 |
| Ridge | 18.01 ± 4.64 |
| Random Forest | 16.40 ± 4.79 |
| **Gradient Boosting** | **14.92 ± 4.90** |

Gradient Boosting was tuned with 3-fold grid search over `n_estimators`, `learning_rate`
and `max_depth`. Best: 400 trees, learning rate 0.1, depth 3.

## 5. Evaluation (untouched test set, 397 rows)
R² = 0.957, MAE = 7.78 lakhs, RMSE = 11.18 lakhs, MAPE = 7.0 %.
See `reports/actual_vs_predicted.png` and `reports/feature_importance.png`.
Residuals are centred on zero; the model under-predicts the most expensive
houses (the high-price tail), a common limitation of tree ensembles.

## 6. Prediction interface
`streamlit run app.py` – users select location/furnishing and enter size, rooms, age, floor,
parking and metro distance. Inputs are validated (`src/predict.py`) and the app shows the
estimated price in lakhs/crore and price per sq.ft.

## 7. Testing
14 automated pytest tests cover: duplicate/invalid-row removal, missing-value handling,
unseen categories, minimum R² on a hold-out set (> 0.85), monotonic sanity checks (bigger
area ⇒ higher price; prime location ⇒ higher price), input validation (bad/missing values),
and metric correctness. The Streamlit app was also smoke-tested headlessly.

## 8. Limitations and future work
- Synthetic data; validate on a real dataset (e.g. Kaggle Bengaluru house prices).
- Add features such as lat/long, amenities, builder; try XGBoost/LightGBM.
- Add prediction intervals (quantile regression) and model monitoring.
- Deploy on Streamlit Community Cloud or a Docker container.
