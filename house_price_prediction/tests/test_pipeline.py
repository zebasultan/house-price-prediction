import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src import config as C
from src.evaluate import regression_metrics
from src.generate_data import generate
from src.predict import predict_price, validate_input
from src.preprocess import build_preprocessor, clean_data
from src.train import make_pipeline

VALID = {"area_sqft": 1200, "bedrooms": 3, "bathrooms": 2, "age_years": 5,
         "parking": 1, "floor": 4, "distance_to_metro_km": 2.0,
         "location": "Whitefield", "furnishing": "Semi-Furnished"}


@pytest.fixture(scope="module")
def df():
    return clean_data(generate(n=800))


@pytest.fixture(scope="module")
def model(df):
    X_tr, _, y_tr, _ = train_test_split(df[C.FEATURES], df[C.TARGET],
                                        test_size=.2, random_state=0)
    return make_pipeline(GradientBoostingRegressor(random_state=0)).fit(X_tr, y_tr)


# ---- preprocessing ----
def test_clean_removes_duplicates_and_invalid_prices():
    raw = generate(n=800)
    clean = clean_data(raw)
    assert len(clean) < len(raw)
    assert clean.duplicated().sum() == 0
    assert (clean[C.TARGET] > 0).all()


def test_preprocessor_handles_missing_values(df):
    X = df[C.FEATURES].copy()
    X.iloc[0, X.columns.get_loc("area_sqft")] = np.nan
    X.iloc[1, X.columns.get_loc("location")] = np.nan
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()


def test_preprocessor_ignores_unseen_category(df):
    prep = build_preprocessor().fit(df[C.FEATURES])
    new = df[C.FEATURES].head(1).copy()
    new["location"] = "Atlantis"
    prep.transform(new)  # must not raise


# ---- model ----
def test_model_quality_on_holdout(df):
    X_tr, X_te, y_tr, y_te = train_test_split(df[C.FEATURES], df[C.TARGET],
                                              test_size=.2, random_state=1)
    m = make_pipeline(GradientBoostingRegressor(random_state=0)).fit(X_tr, y_tr)
    assert regression_metrics(y_te, m.predict(X_te))["R2"] > 0.85


def test_prediction_is_positive_and_sensible(model):
    assert 10 < predict_price(VALID, model) < 1000


def test_bigger_house_costs_more(model):
    assert predict_price({**VALID, "area_sqft": 2500}, model) > \
           predict_price({**VALID, "area_sqft": 700}, model)


def test_prime_location_costs_more(model):
    assert predict_price({**VALID, "location": "Indiranagar"}, model) > \
           predict_price({**VALID, "location": "Electronic City"}, model)


# ---- input validation ----
@pytest.mark.parametrize("bad", [
    {"area_sqft": -5}, {"bedrooms": 99}, {"location": "Mars"},
    {"furnishing": "Lava"}, {"age_years": "abc"},
])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        validate_input({**VALID, **bad})


def test_missing_field_rejected():
    data = dict(VALID)
    data.pop("floor")
    with pytest.raises(ValueError):
        validate_input(data)


# ---- metrics ----
def test_metrics_perfect_prediction():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["MAE"] == 0 and m["R2"] == 1
