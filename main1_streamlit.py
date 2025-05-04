import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.title("Commodity Price Prediction Dashboard")

# 1) Load data
try:
    df = pd.read_csv("new.csv")
except FileNotFoundError:
    st.error("Could not find 'new.csv'. Please upload it alongside this script.")
    st.stop()

# 2) Normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"_x0020_", "_", regex=True)
      .str.replace(r"\s+", "_", regex=True)
)

# 3) Fill missing numeric values
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 4) Ensure Day_Number exists
if 'Day_Number' not in df.columns:
    if 'Arrival_Date' in df.columns:
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
        base = df['Arrival_Date'].min()
        df['Day_Number'] = (df['Arrival_Date'] - base).dt.days.fillna(0).astype(int)
    else:
        st.error("Your data needs either a 'Day_Number' or an 'Arrival_Date' column.")
        st.stop()

# 5) Normalize & encode Commodity
df['Commodity'] = df['Commodity'].astype(str).str.strip().str.title()
le = LabelEncoder()
df['Commodity_Encoded'] = le.fit_transform(df['Commodity'])

# 6) Select numeric features only
FEATURES = ['Day_Number', 'Min_Price', 'Max_Price']
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing required numeric columns: {missing}")
    st.stop()

X = df[FEATURES].values
y = df['Modal_Price'].values

# 7) Model choice
model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=50, random_state=42)
else:
    model = LinearRegression()

# 8) Train
model.fit(X, y)

# 9) User picks commodity
commodity = st.selectbox("Commodity", df['Commodity'].unique())

# 10) Filter latest row
code = le.transform([commodity])[0]
sub = df[df['Commodity_Encoded'] == code]
if sub.empty:
    st.error(f"No data for {commodity}")
    st.stop()
latest = sub.iloc[-1]

# 11) Forecast
intervals = [0, 3, 7, 15]
labels    = ["Today", "In 3 days", "Next week", "Next 15 days"]
forecasts = {}
for days, lbl in zip(intervals, labels):
    vals = np.array([[latest['Day_Number'] + days,
                      latest['Min_Price'],
                      latest['Max_Price']]])
    pred = model.predict(vals)[0]
    forecasts[lbl] = round(pred, 2)

# 12) Display
st.subheader(f"Results for {commodity}")
for lbl, price in forecasts.items():
    st.write(f"- **{lbl}**: {price}")

# 13) Plot
fig, ax = plt.subplots()
ax.plot(labels, list(forecasts.values()), marker='o')
ax.set_title(f"{commodity} Price Forecast")
ax.set_ylabel("Price")
st.pyplot(fig)
