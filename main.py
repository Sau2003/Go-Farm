from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

app = FastAPI()

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

# Request model
class CommodityRequest(BaseModel):
    commodity: str

@app.post("/predict")
def predict_price(request: CommodityRequest):
    # Normalize user input
    commodity_input = request.commodity.strip().title()
    
    # Debug: Print available commodities
    print(f"User input: '{commodity_input}'")
    print("Available commodities:", list(df['Commodity'].unique()))

    # Check if commodity exists
    if commodity_input not in df['Commodity'].values:
        return {"error": f"Commodity '{commodity_input}' not found. Check spelling!"}

    # Transform commodity input
    commodity_code = label_encoders['Commodity'].transform([commodity_input])[0]
    
    # Debug: Print encoded commodity values
    print(f"Encoded value for '{commodity_input}': {commodity_code}")
    print("Unique encoded commodities:", df['Commodity_Encoded'].unique())

    # Filter dataset for selected commodity
    commodity_data = df[df['Commodity_Encoded'] == commodity_code]

    # Debug: Print number of rows found
    print(f"Rows found for '{commodity_input}': {len(commodity_data)}")

    if commodity_data.empty:
        return {"error": f"No data available for commodity '{commodity_input}'."}

    # Get the latest entry
    latest_entry = commodity_data.iloc[-1].copy()

    # Ensure required features exist
    if 'Day_Number' not in latest_entry:
        return {"error": "Feature 'Day_Number' missing in dataset."}

    # Forecast price intervals
    future_days = [0, 3, 7, 15]
    forecast_labels = ["Today", "In 3 days", "Next week", "Next 15 days"]
    forecasted_prices = {}

    for days, label in zip(future_days, forecast_labels):
        latest_entry['Day_Number'] += days  # Adjust date feature
        
        # Debug: Print features before prediction
        print(f"Predicting for {label}: {latest_entry[feature_columns].values}")
        
        future_price = lr_model.predict([latest_entry[feature_columns]])[0]
        forecasted_prices[label] = round(future_price, 2)

    return {
        "current_price": forecasted_prices["Today"],
        "forecast": forecasted_prices
    }
