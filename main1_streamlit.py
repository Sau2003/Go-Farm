import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv("new.csv")
df.columns = df.columns.str.strip().str.replace(r"_x0020_", "_", regex=True)

st.write("Data columns:", df.columns.tolist())

df['Commodity'] = df['Commodity'].str.strip().str.title()
label_encoders = {"Commodity": LabelEncoder()}
df['Commodity_Encoded'] = label_encoders["Commodity"].fit_transform(df['Commodity'])

feature_columns = [c for c in df.columns if c not in ['Modal_Price', 'Commodity']]
X = df[feature_columns]
y = df['Modal_Price']

lr_model = LinearRegression()
lr_model.fit(X, y)

st.title("Commodity Price Prediction")
st.write("Enter a commodity name to see its current price and forecasts.")

commodity_input = st.text_input("Commodity Name:")

if commodity_input:
    ci = commodity_input.strip().title()
    if ci not in df['Commodity'].values:
        st.error(f"Commodity '{ci}' not found. Please check spelling.")
    else:
        code = label_encoders['Commodity'].transform([ci])[0]
        data_i = df[df['Commodity_Encoded'] == code]
        if data_i.empty:
            st.error(f"No data available for '{ci}'.")
        else:
            latest = data_i.iloc[-1].copy()
            future_days = [0, 3, 7, 15]
            forecast_tags = ["Today", "In 3 days", "Next week", "Next 15 days"]
            forecasts = {}
            for days, tag in zip(future_days, forecast_tags):
                latest['Day_Number'] += days
                price = lr_model.predict([latest[feature_columns]])[0]
                forecasts[tag] = round(price, 2)

            st.subheader(f"Results for {ci}")
            st.write(f"**Current Price:** {forecasts['Today']}")
            st.write("**Forecasts:**")
            for tag, val in forecasts.items():
                st.write(f"- {tag}: {val}")
