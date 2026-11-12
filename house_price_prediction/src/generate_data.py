"""Generate a realistic synthetic housing dataset (Bengaluru-style).

Replace data/housing.csv with any real dataset (e.g. from Kaggle) that has the
same columns and the rest of the pipeline works unchanged.
"""
import numpy as np
import pandas as pd
from src.config import DATA_PATH, LOCATIONS, FURNISHING, RANDOM_STATE

# base price per sq.ft (INR) by location
LOC_RATE = {
    "Indiranagar": 14000, "Koramangala": 13000, "Jayanagar": 11500,
    "HSR Layout": 10500, "Whitefield": 8500, "Hebbal": 8000,
    "Electronic City": 6000, "Yelahanka": 6500,
}
FURN_BONUS = {"Unfurnished": 0.0, "Semi-Furnished": 0.04, "Furnished": 0.09}


def generate(n: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    location = rng.choice(LOCATIONS, n)
    furnishing = rng.choice(FURNISHING, n, p=[0.35, 0.40, 0.25])
    bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.38, 0.32, 0.14, 0.04])
    area = (bedrooms * rng.normal(430, 60, n) + rng.normal(150, 80, n)).clip(350, 5000)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 6)
    age = rng.integers(0, 31, n)
    parking = rng.choice([0, 1, 2], n, p=[0.2, 0.55, 0.25])
    floor = rng.integers(0, 21, n)
    metro = rng.gamma(2.0, 1.5, n).clip(0.1, 15).round(2)

    rate = np.array([LOC_RATE[l] for l in location], dtype=float)
    price = area * rate / 1e5                                  # in lakhs
    price *= 1 - 0.012 * age                                   # depreciation
    price *= 1 + np.array([FURN_BONUS[f] for f in furnishing])
    price *= 1 - 0.015 * np.minimum(metro, 10)                 # metro proximity
    price += parking * 4 + floor * 0.35 + bathrooms * 2
    price *= rng.normal(1, 0.06, n)                            # market noise
    df = pd.DataFrame({
        "area_sqft": area.round(0), "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age, "parking": parking, "floor": floor,
        "distance_to_metro_km": metro, "location": location,
        "furnishing": furnishing, "price_lakhs": price.round(2),
    })

    # --- inject real-world mess so preprocessing has work to do ---
    for col, frac in [("area_sqft", .03), ("age_years", .03),
                      ("furnishing", .02), ("distance_to_metro_km", .02)]:
        df.loc[df.sample(frac=frac, random_state=seed).index, col] = np.nan
    df.loc[df.sample(n=8, random_state=1).index, "price_lakhs"] *= 6      # outliers
    df.loc[df.sample(n=5, random_state=2).index, "price_lakhs"] = -10     # invalid
    df = pd.concat([df, df.sample(15, random_state=3)], ignore_index=True)  # duplicates
    return df


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(DATA_PATH, index=False)
    print(f"Saved {DATA_PATH}")
