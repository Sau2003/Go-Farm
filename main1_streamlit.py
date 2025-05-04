import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("new.csv")  # Replace with actual dataset path

# Normalize commodity names in dataset
df['Commodity'] = df['Commodity'].str.strip().str.title()  # Ensure consistency

# Encode Commodity column
label_encoders = {"Commodity": LabelEncoder()}
df['Commodity_Encoded'] = label_encoders["Commodity"].fit_transform(df['Commodity'])

# Define feature columns (Ensure 'Day_Number' exists)
feature_columns = [col for col in df.columns if col not in ['Modal_Price', 'Commodity']]
X = df[feature_columns]
y = df['Modal_Price']

# Train the model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Streamlit UI
st.title("Commodity Price Prediction")
st.write("This app predicts the commodity price for a selected commodity and forecasts future prices.")

# Input for commodity
commodity_input = st.text_input("Enter Commodity Name:")

if commodity_input:
    # Normalize user input
    commodity_input = commodity_input.strip().title()

    # Check if commodity exists in dataset
    if commodity_input not in df['Commodity'].values:
        st.error(f"Commodity '{commodity_input}' not found. Check spelling!")
    else:
        # Transform commodity input
        commodity_code = label_encoders['Commodity'].transform([commodity_input])[0]

        # Filter dataset for selected commodity
        commodity_data = df[df['Commodity_Encoded'] == commodity_code]

        if commodity_data.empty:
            st.error(f"No data available for commodity '{commodity_input}'.")
        else:
            # Get the latest entry
            latest_entry = commodity_data.iloc[-1].copy()

            # Forecast price intervals
            future_days = [0, 3, 7, 15]
            forecast_labels = ["Today", "In 3 days", "Next week", "Next 15 days"]
            forecasted_prices = {}

            for days, label in zip(future_days, forecast_labels):
                latest_entry['Day_Number'] += days  # Adjust date feature

                # Predict price for the adjusted feature
                future_price = lr_model.predict([latest_entry[feature_columns]])[0]
                forecasted_prices[label] = round(future_price, 2)

            # Display results
            st.write(f"**Current Price for {commodity_input}:** {forecasted_prices['Today']}")

            st.write("**Price Forecasts:**")
            for label, price in forecasted_prices.items():
                st.write(f"{label}: {price}")
