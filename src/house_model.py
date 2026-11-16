import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF",
           "YearBuilt", "FullBath", "BedroomAbvGr"]
CATEGORICAL = ["Neighborhood"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path="data/train.csv"):
    return pd.read_csv(path)


def make_model():
    prep = ColumnTransformer(
        [("hood", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([
        ("prep", prep),
        ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                         max_depth=3, random_state=42)),
    ])


def evaluate(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    model = make_model().fit(train[FEATURES], train["SalePrice"])
    pred = model.predict(test[FEATURES])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(make_model(), df[FEATURES], df["SalePrice"],
                              cv=cv, scoring="neg_mean_absolute_error")
    return {
        "test_mae": round(mean_absolute_error(test["SalePrice"], pred)),
        "test_r2": round(r2_score(test["SalePrice"], pred), 2),
        "cv_mae": round(scores.mean()),
    }


def save_model(df, results, out_dir="app"):
    final = make_model().fit(df[FEATURES], df["SalePrice"])
    out = Path(out_dir)
    joblib.dump(final, out / "house_price_model.joblib")
    meta = {
        "features": FEATURES,
        "neighborhoods": sorted(df["Neighborhood"].unique().tolist()),
        "ranges": {c: [int(df[c].min()), int(df[c].max())] for c in NUMERIC},
        "test_mae": results["test_mae"],
        "cv_mae": results["cv_mae"],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }
    with open(out / "model_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    return final
def check_ranges(house, ranges):
    """Raise ValueError if a number is outside the range seen in training."""
    for name, (low, high) in ranges.items():
        value = house[name]
        if value < low or value > high:
            raise ValueError(f"{name} = {value} is outside the range {low} to {high}")