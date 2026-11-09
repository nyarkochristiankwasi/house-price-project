import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Ames House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_model():
    return joblib.load(HERE / "house_price_model.joblib")


@st.cache_data
def load_info():
    with open(HERE / "model_info.json") as f:
        return json.load(f)


model = load_model()
info = load_info()
r = info["ranges"]
typical_error = info["cv_mae"]

st.title("🏠 House price predictor")
st.write("Enter a few details about a house to get an estimated sale price. "
         "The model was trained on 1,460 house sales in Ames, Iowa.")

col1, col2 = st.columns(2)
with col1:
    living_area = st.number_input("Living area (sq ft)", r["GrLivArea"][0], r["GrLivArea"][1], 1800, step=50)
    quality = st.slider("Overall quality (1 = poor, 10 = excellent)", r["OverallQual"][0], r["OverallQual"][1], 7)
    garage = st.slider("Garage size (cars)", r["GarageCars"][0], r["GarageCars"][1], 2)
    basement = st.number_input("Basement area (sq ft)", r["TotalBsmtSF"][0], r["TotalBsmtSF"][1], 900, step=50)
with col2:
    year_built = st.number_input("Year built", r["YearBuilt"][0], r["YearBuilt"][1], 2000, step=1)
    baths = st.slider("Full bathrooms", r["FullBath"][0], r["FullBath"][1], 2)
    bedrooms = st.slider("Bedrooms", r["BedroomAbvGr"][0], r["BedroomAbvGr"][1], 3)
    default_hood = info["neighborhoods"].index("CollgCr") if "CollgCr" in info["neighborhoods"] else 0
    hood = st.selectbox("Neighborhood", info["neighborhoods"], index=default_hood)

if st.button("Predict price", type="primary"):
    house = pd.DataFrame([{
        "GrLivArea": living_area,
        "OverallQual": quality,
        "GarageCars": garage,
        "TotalBsmtSF": basement,
        "YearBuilt": year_built,
        "FullBath": baths,
        "BedroomAbvGr": bedrooms,
        "Neighborhood": hood,
    }])
    price = float(model.predict(house[info["features"]])[0])
    low = max(price - typical_error, 0)
    high = price + typical_error

    st.metric("Estimated price", f"${price:,.0f}")
    st.write(f"Likely range: **${low:,.0f} to ${high:,.0f}**")
    st.caption(
        f"The range is the estimate plus or minus the model's typical error "
        f"(about ${typical_error:,.0f} on houses it had not seen)."
    )

st.divider()
st.caption(
    "This is a learning project trained on house sales in Ames, Iowa. "
    "It is not valuation advice, and it will not be accurate for other markets."
)