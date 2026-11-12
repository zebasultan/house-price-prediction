"""Data loading, cleaning and the feature-transformation pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def load_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that must happen BEFORE the train/test split.

    1. drop exact duplicates
    2. drop rows with missing / non-positive target
    3. drop physically impossible rows (area <= 0)
    4. remove target outliers with the 1.5*IQR rule on price per sq.ft
       (price-per-sqft is more robust than raw price, which scales with size)
    """
    df = df.drop_duplicates().copy()
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    df = df[df["area_sqft"].isna() | (df["area_sqft"] > 0)]

    ppsf = df[TARGET] * 1e5 / df["area_sqft"]
    q1, q3 = ppsf.quantile([0.25, 0.75])
    iqr = q3 - q1
    keep = ppsf.isna() | ppsf.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df[keep].reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numbers, imputation + one-hot for categories.

    Fitted only on training data inside the model Pipeline -> no data leakage.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
