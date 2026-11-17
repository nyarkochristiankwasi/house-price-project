import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.house_model import FEATURES, check_ranges, make_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "app" / "house_price_model.joblib"
INFO_PATH = ROOT / "app" / "model_info.json"

EXAMPLE_HOUSE = {
    "GrLivArea": 1800, "OverallQual": 7, "GarageCars": 2, "TotalBsmtSF": 900,
    "YearBuilt": 2000, "FullBath": 2, "BedroomAbvGr": 3, "Neighborhood": "CollgCr",
}


def tiny_data():
    n = 30
    return pd.DataFrame({
        "GrLivArea": [1000 + 50 * i for i in range(n)],
        "OverallQual": [5 + i % 4 for i in range(n)],
        "GarageCars": [i % 3 for i in range(n)],
        "TotalBsmtSF": [500 + 20 * i for i in range(n)],
        "YearBuilt": [1950 + i for i in range(n)],
        "FullBath": [1 + i % 2 for i in range(n)],
        "BedroomAbvGr": [2 + i % 3 for i in range(n)],
        "Neighborhood": ["A" if i % 2 else "B" for i in range(n)],
        "SalePrice": [100000 + 3000 * i for i in range(n)],
    })


def test_saved_model_loads():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_prediction_for_a_typical_house_is_sensible():
    model = joblib.load(MODEL_PATH)
    price = float(model.predict(pd.DataFrame([EXAMPLE_HOUSE])[FEATURES])[0])
    assert 100_000 < price < 400_000


def test_out_of_range_input_is_rejected():
    with open(INFO_PATH) as f:
        ranges = json.load(f)["ranges"]
    check_ranges(EXAMPLE_HOUSE, ranges)          # a normal house passes
    too_big = dict(EXAMPLE_HOUSE, GrLivArea=99_999)
    with pytest.raises(ValueError):
        check_ranges(too_big, ranges)


def test_make_model_trains_on_a_small_dataset():
    df = tiny_data()
    model = make_model().fit(df[FEATURES], df["SalePrice"])
    preds = model.predict(df[FEATURES])
    assert len(preds) == len(df)
    assert (preds > 0).all()

def test_model_info_matches_the_code():
    with open(INFO_PATH) as f:
        info = json.load(f)
    for key in ["features", "neighborhoods", "ranges", "cv_mae"]:
        assert key in info
    assert info["features"] == FEATURES