"""Streamlit prediction interface.  Run:  streamlit run app.py"""
import streamlit as st
from src import config as C
from src.predict import load_model, predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.caption("ML model trained on housing data · prices in INR")


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", C.LOCATIONS)
    area = st.number_input("Area (sq.ft)", 200, 10000, 1200, step=50)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
with col2:
    furnishing = st.selectbox("Furnishing", C.FURNISHING, index=1)
    age = st.slider("Age of property (years)", 0, 50, 5)
    floor = st.slider("Floor", 0, 40, 4)
    parking = st.slider("Parking spots", 0, 5, 1)
    metro = st.number_input("Distance to metro (km)", 0.0, 50.0, 2.0, step=0.5)

if st.button("Predict price", type="primary"):
    try:
        lakhs = predict_price({
            "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "age_years": age, "parking": parking, "floor": floor,
            "distance_to_metro_km": metro, "location": location,
            "furnishing": furnishing}, model)
        st.success(f"Estimated price: **₹{lakhs:,.1f} lakhs** (≈ ₹{lakhs/100:,.2f} crore)")
        st.caption(f"≈ ₹{lakhs*1e5/area:,.0f} per sq.ft. This is an estimate, not a valuation.")
    except ValueError as e:
        st.error(str(e))
