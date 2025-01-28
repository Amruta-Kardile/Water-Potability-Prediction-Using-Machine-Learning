import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the combined model and encoder
with open('food_product_model.pkl', 'rb') as file:
    data = pickle.load(file)

model = data['model']
encoder = data['encoder']

# Feature names (adjust based on dataset columns)
feature_names = [
    'Land use change', 'Animal Feed', 'Farm', 'Processing', 'Transport',
    'Packging', 'Retail', 'Total_emissions',
    'Eutrophying emissions per kilogram (gPO₄eq per kilogram)',
    'Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)'
]

# Streamlit app layout
st.set_page_config(page_title="Food Product Predictor", layout="wide")
st.title("🌾 Food Product Prediction App")
st.write(
    """
    Welcome to the Food Product Prediction App! 🎉  
    Use the sliders below to input feature values, and the app will predict the food product based on the data.  
    Additionally, explore the feature importance analysis to understand how the model makes predictions.
    """
)

# Sliders for input
st.sidebar.header("Input Features")
inputs = []
for feature in feature_names:
    value = st.sidebar.slider(
        f"{feature}:",
        min_value=0.0, max_value=100.0, value=0.0, step=0.1
    )
    inputs.append(value)

# Main page: Prediction
st.header("🎯 Prediction")
if st.button("Predict"):
    # Convert inputs to a NumPy array
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform(prediction)
    st.success(f"Predicted Food Product: **{predicted_label[0]}**")

# Main page: Feature importance analysis
st.header("📊 Feature Importance Analysis")
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display importance as a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Feature Importance for Food Product Prediction')
st.pyplot(fig)

# Display importance data as a table
st.subheader("Feature Importance Table")
st.dataframe(importance_df)
