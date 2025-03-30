from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working!"}

# Load dataset
file_path = "new.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found.")

df = pd.read_csv(file_path)

# Fix column names if needed
df.columns = df.columns.str.replace("_x0020_", "_")
df.dropna(inplace=True)

# Convert Arrival Date to datetime
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
df.sort_values(by=['Arrival_Date'], inplace=True)

# Convert categorical variables to numerical
label_encoders = {}
categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for future use

# Generate 'Day_Number' based on days since the first date in the dataset
df['Day_Number'] = (df['Arrival_Date'] - df['Arrival_Date'].min()).dt.days

# Features and target variable
X = df[['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Day_Number', 'Min_Price', 'Max_Price']]
y = df['Modal_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Pydantic model for request body
class CommodityRequest(BaseModel):
    commodity: str

@app.post("/predict")
def predict_price(request: CommodityRequest):
    # Normalize input
    commodity_input = request.commodity.strip().title()

    # Debugging: Print user input and available labels
    print(f"User input commodity: '{commodity_input}'")

    # Check if the commodity exists in the dataset
    if commodity_input not in label_encoders['Commodity'].classes_:
        return {"error": f"Commodity '{commodity_input}' not found in the dataset."}

    # Encode commodity input
    commodity_code = label_encoders['Commodity'].transform([commodity_input])[0]

    # Filter dataset for the commodity
    commodity_data = df[df['Commodity'] == commodity_code]

    if commodity_data.empty:
        return {"error": f"No data available for commodity '{commodity_input}'."}

    # Get the latest available data for the commodity
    latest_entry = commodity_data.iloc[-1].copy()

    # Ensure features are correctly formatted
    feature_columns = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Day_Number', 'Min_Price', 'Max_Price']

    # Ensure 'Day_Number' is an integer
    latest_entry['Day_Number'] = int(latest_entry['Day_Number'])

    # Forecast price intervals
    future_days = [0, 3, 7, 15]
    forecast_labels = ["Today", "In 3 days", "Next week", "Next 15 days"]
    forecasted_prices = {}

    for days, label in zip(future_days, forecast_labels):
        latest_entry['Day_Number'] += days  # Adjust date feature
        
        # Ensure all features are numeric before passing them to the model
        numeric_features = latest_entry[feature_columns].astype(float)

        # Predict price
        future_price = lr_model.predict([numeric_features])[0]

        forecasted_prices[label] = round(future_price, 2)

    return {
        "current_price": forecasted_prices["Today"],
        "forecast": forecasted_prices
    }
