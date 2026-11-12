"""Central configuration: paths, column names, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORTS_DIR = ROOT / "reports"

TARGET = "price_lakhs"          # price in INR lakhs (1 lakh = 100,000 INR)
NUMERIC_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "age_years",
    "parking", "floor", "distance_to_metro_km",
]
CATEGORICAL_FEATURES = ["location", "furnishing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LOCATIONS = [
    "Whitefield", "Koramangala", "Indiranagar", "HSR Layout",
    "Electronic City", "Hebbal", "Jayanagar", "Yelahanka",
]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
