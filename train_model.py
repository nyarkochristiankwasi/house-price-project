import json
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/train.csv")

numeric = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF",
           "YearBuilt", "FullBath", "BedroomAbvGr"]
categorical = ["Neighborhood"]
features = numeric + categorical


def make_model():
    prep = ColumnTransformer(
        [("hood", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    return Pipeline([
        ("prep", prep),
        ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                         max_depth=3, random_state=42)),
    ])


# 1. Check accuracy on houses the model has not seen
train, test = train_test_split(df, test_size=0.2, random_state=42)
check = make_model().fit(train[features], train["SalePrice"])
pred = check.predict(test[features])
test_mae = round(mean_absolute_error(test["SalePrice"], pred))
print("Test MAE:", test_mae, "| R2:", round(r2_score(test["SalePrice"], pred), 2))

# 2. A firmer estimate with 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -cross_val_score(make_model(), df[features], df["SalePrice"],
                             cv=cv, scoring="neg_mean_absolute_error")
cv_mae = round(cv_scores.mean())
print("Cross-validated MAE:", cv_mae)

# 3. Train the final model on all 1,460 houses and save it
final = make_model().fit(df[features], df["SalePrice"])
joblib.dump(final, "app/house_price_model.joblib")

meta = {
    "features": features,
    "neighborhoods": sorted(df["Neighborhood"].unique().tolist()),
    "ranges": {c: [int(df[c].min()), int(df[c].max())] for c in numeric},
    "test_mae": test_mae,
    "cv_mae": cv_mae,
    "sklearn_version": sklearn.__version__,
    "pandas_version": pd.__version__,
}
with open("app/model_info.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved app/house_price_model.joblib and app/model_info.json")
print("scikit-learn", sklearn.__version__, "| pandas", pd.__version__)