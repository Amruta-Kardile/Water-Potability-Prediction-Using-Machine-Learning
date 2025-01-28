import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler from the .pkl file
with open("model_and_scaler.pkl", "rb") as file:
    data = pickle.load(file)
    model = data["model"]
    scaler = data["scaler"]

# Add a custom background image
def add_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('/water.jpeg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_background_image()

# App title and description
st.title("🌊 **Water Potability Prediction** 🌊")
st.markdown("""
Welcome to the **Water Potability Predictor**! 💧  
This app helps you determine whether water is **safe for drinking** based on its chemical properties.  
Simply adjust the parameters on the **left sidebar**, click **Predict**, and get instant results! 🚀
""")

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar for input parameters
st.sidebar.header("🔧 **Adjust Water Quality Parameters:**")
ph = st.sidebar.slider("📏 **pH Level**", 0.0, 14.0, 7.0, step=0.1, help="Measure of how acidic or basic the water is (0-14).")
hardness = st.sidebar.slider("💧 **Hardness (mg/L)**", 0.0, 500.0, 200.0, step=1.0, help="Amount of dissolved calcium and magnesium in water.")
solids = st.sidebar.number_input("🔬 **Solids (ppm)**", 0.0, 50000.0, 20000.0, step=100.0, help="Total dissolved solids in the water.")
chloramines = st.sidebar.slider("🧪 **Chloramines (mg/L)**", 0.0, 15.0, 7.0, step=0.1, help="Residual disinfectant in drinking water.")
sulfate = st.sidebar.number_input("🌡️ **Sulfate (mg/L)**", 0.0, 500.0, 333.0, step=1.0, help="Sulfur compounds dissolved in water.")
conductivity = st.sidebar.slider("⚡ **Conductivity (μS/cm)**", 0.0, 800.0, 400.0, step=1.0, help="Water's ability to conduct electricity.")
organic_carbon = st.sidebar.slider("🌿 **Organic Carbon (mg/L)**", 0.0, 30.0, 15.0, step=0.1, help="Measure of organic compounds in water.")
trihalomethanes = st.sidebar.slider("🧴 **Trihalomethanes (μg/L)**", 0.0, 120.0, 60.0, step=1.0, help="Byproduct of water disinfection.")
turbidity = st.sidebar.slider("🌫️ **Turbidity (NTU)**", 0.0, 10.0, 4.0, step=0.1, help="Cloudiness of water caused by particles.")

# Derived features
hardness_by_conductivity = hardness / (conductivity + 1e-5)
organic_carbon_ratio = organic_carbon / (solids + 1e-5)
chloramines_per_turbidity = chloramines / (turbidity + 1e-5)

# Create input feature array
input_features = np.array([
    ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon,
    trihalomethanes, turbidity, hardness_by_conductivity,
    organic_carbon_ratio, chloramines_per_turbidity
]).reshape(1, -1)

# Add Predict button
if st.button("🚀 **Predict Water Potability**"):
    # Scale the input features
    input_features_scaled = scaler.transform(input_features)
    
    # Predict using the model
    prediction = model.predict(input_features_scaled)
    
    # Display results with emoji-based output
    st.markdown("### Prediction Result:")
    if prediction[0] == 1:
        st.success("🚰 **The water is Potable!** It is safe for drinking. 🥤")
        st.balloons()
    else:
        st.error("❌ **The water is Not Potable!** It is unsafe for drinking. 💀")

# Footer
st.markdown("""
---
👨‍🔬 *Built with ❤️ by your friendly data scientist!*  
💡 **Tip**: Ensure accurate readings of your water properties for best results!
""")
