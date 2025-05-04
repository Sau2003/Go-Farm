import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.title("Commodity Price Prediction Dashboard")

# 1) Load data with error handling
try:
    df = pd.read_csv("new.csv")
except FileNotFoundError:
    st.error("Could not find 'new.csv'. Please upload it to the same folder as this script.")
    st.stop()

# 2) Normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"_x0020_", "_", regex=True)
      .str.replace(r"\s+", "_", regex=True)
)

# 3) Show the columns for debugging
st.write("Columns:", df.columns.tolist())

# 4) Handle missing values
if df.isnull().any().any():
    st.warning("Filling missing values with column means.")
    df.fillna(df.mean(numeric_only=True), inplace=True)

# 5) Preprocess commodity column
df['Commodity'] = df['Commodity'].astype(str).str.strip().str.title()
le = LabelEncoder()
df['Commodity_Encoded'] = le.fit_transform(df['Commodity'])

# 6) Verify 'Day_Number' exists or create from date
if 'Day_Number' not in df.columns:
    if 'Arrival_Date' in df.columns:
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
        min_date = df['Arrival_Date'].min()
        df['Day_Number'] = (df['Arrival_Date'] - min_date).dt.days.fillna(0).astype(int)
    else:
        st.error("Dataset lacks both 'Day_Number' and 'Arrival_Date'.")
        st.stop()

# 7) Define features and target
FEATURES = [c for c in df.columns if c not in ['Modal_Price', 'Commodity', 'Arrival_Date']]
X = df[FEATURES]
y = df['Modal_Price']

# 8) Model selection
model_choice = st.sidebar.radio("Select model", ["Linear Regression", "Random Forest"])
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=50, random_state=42)
else:
    model = LinearRegression()

model.fit(X, y)

# 9) User selects commodity
commodity = st.selectbox("Choose a Commodity", df['Commodity'].unique())

# 10) Optional custom date
custom_date = st.date_input("Predict price for date", date.today())
date_diff = (custom_date - df['Arrival_Date'].min().date()).days

# 11) Filter latest record for that commodity
code = le.transform([commodity])[0]
subset = df[df['Commodity_Encoded'] == code]
if subset.empty:
    st.error(f"No data for {commodity}")
    st.stop()
latest = subset.iloc[-1].copy()

# 12) Forecast prices
intervals = [0, 3, 7, 15, date_diff]
labels    = ["Today", "In 3 days", "Next week", "Next 15 days", f"On {custom_date}"]
forecasts = {}

for days, label in zip(intervals, labels):
    entry = latest.copy()
    entry['Day_Number'] = entry['Day_Number'] + days
    pred = model.predict([entry[FEATURES]])[0]
    forecasts[label] = round(pred, 2)

# 13) Display results
st.subheader(f"Predictions for {commodity}")
for label, price in forecasts.items():
    st.write(f"- **{label}**: {price}")

# 14) Plot forecast curve
fig, ax = plt.subplots()
ax.plot(list(forecasts.keys()), list(forecasts.values()), marker='o')
ax.set_title(f"Price Forecast: {commodity}")
ax.set_ylabel("Price")
ax.set_xticklabels(list(forecasts.keys()), rotation=30)
st.pyplot(fig)
