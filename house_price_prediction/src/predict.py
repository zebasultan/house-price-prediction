"""Load the trained model and predict a price for one house."""
import joblib
import pandas as pd
from src import config as C

# (min, max) sanity ranges for user input validation
LIMITS = {
    "area_sqft": (200, 10000), "bedrooms": (1, 10), "bathrooms": (1, 10),
    "age_years": (0, 100), "parking": (0, 10), "floor": (0, 60),
    "distance_to_metro_km": (0, 50),
}


def validate_input(data: dict) -> dict:
    """Raise ValueError for missing, out-of-range or unknown values."""
    missing = [f for f in C.FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    for col, (lo, hi) in LIMITS.items():
        try:
            val = float(data[col])
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number")
        if not lo <= val <= hi:
            raise ValueError(f"{col} must be between {lo} and {hi}")
    if data["location"] not in C.LOCATIONS:
        raise ValueError(f"location must be one of {C.LOCATIONS}")
    if data["furnishing"] not in C.FURNISHING:
        raise ValueError(f"furnishing must be one of {C.FURNISHING}")
    return data


def load_model(path=C.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_price(data: dict, model=None) -> float:
    """Return the predicted price in INR lakhs."""
    validate_input(data)
    model = model or load_model()
    row = pd.DataFrame([{f: data[f] for f in C.FEATURES}])
    return float(model.predict(row)[0])
