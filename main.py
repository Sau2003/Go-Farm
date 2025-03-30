# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, cross_val_score
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_absolute_error, r2_score

# # # Load dataset
# # file_path = "new.csv"
# # df = pd.read_csv(file_path)

# # # Rename columns for easier access
# # df.columns = df.columns.str.replace("_x0020_", "_")  # Fix column names if needed

# # # Drop missing values
# # df.dropna(inplace=True)

# # # Convert 'Arrival_Date' to datetime
# # df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

# # # Sorting data by date
# # df.sort_values(by=['Arrival_Date'], inplace=True)

# # # Convert categorical variables to numerical
# # label_encoders = {}
# # categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']

# # for col in categorical_cols:
# #     le = LabelEncoder()
# #     df[col] = le.fit_transform(df[col])
# #     label_encoders[col] = le  # Save encoder for future use

# # # Selecting features and target variable
# # df['Day_Number'] = (df['Arrival_Date'] - df['Arrival_Date'].min()).dt.days  # Convert dates to numerical format

# # X = df[['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Day_Number', 'Min_Price', 'Max_Price']]
# # y = df['Modal_Price']  # Target variable

# # # Splitting dataset
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Initialize and train Linear Regression model
# # lr_model = LinearRegression()
# # lr_model.fit(X_train, y_train)

# # # Cross-Validation to check for overfitting
# # cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
# # print(f"Cross-Validation R-squared Scores: {cv_scores}")
# # print(f"Mean CV R-squared Score: {np.mean(cv_scores)}")

# # # Predictions for testing data
# # y_pred = lr_model.predict(X_test)

# # # Evaluation
# # mae = mean_absolute_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)
# # accuracy = (1 - mae / np.mean(y_test)) * 100

# # print(f"Mean Absolute Error: {mae}")
# # print(f"R-squared Score: {r2}")
# # print(f"Model Accuracy: {accuracy:.2f}%")

# # # User selects a commodity
# # user_commodity = input("Enter the Commodity Name: ").title()

# # # Encode user input to match the dataset encoding
# # if user_commodity in label_encoders['Commodity'].classes_:
# #     encoded_commodity = label_encoders['Commodity'].transform([user_commodity])[0]
# # else:
# #     print("Commodity not found. Please check spelling.")
# #     exit()

# # # Filter dataset for selected commodity
# # commodity_data = df[df['Commodity'] == encoded_commodity]

# # if commodity_data.empty:
# #     print("No data available for this commodity.")
# #     exit()

# # # Get the latest available data for the commodity
# # latest_entry = commodity_data.iloc[-1].copy()

# # # Display current modal price
# # current_price = latest_entry['Modal_Price']
# # print(f"\nCurrent Modal Price for {user_commodity}: {current_price:.2f}")

# # # Forecast price intervals
# # future_days = [0, 3, 7, 15]  # Updated intervals
# # forecast_labels = ["Today", "In 3 days", "Next week", "Next 15 days"]
# # forecasted_prices = {}

# # for days, label in zip(future_days, forecast_labels):
# #     latest_entry['Day_Number'] += days  # Adjust date feature
# #     future_price = lr_model.predict([latest_entry[X.columns]])[0]  # Predict price
# #     forecasted_prices[label] = future_price

# # # Display forecasted prices
# # print("\nForecasted Prices:")
# # for key, value in forecasted_prices.items():
# #     print(f"{key}: {value:.2f}")




# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression

# app = FastAPI()

# # Load dataset
# df = pd.read_csv("new.csv")  # Replace with actual dataset path

# # Normalize commodity names in dataset
# df['Commodity'] = df['Commodity'].str.strip().str.title()  # Ensure consistency

# # Encode Commodity column
# label_encoders = {"Commodity": LabelEncoder()}
# df['Commodity_Encoded'] = label_encoders["Commodity"].fit_transform(df['Commodity'])

# # Define feature columns (Ensure 'Day_Number' exists)
# feature_columns = [col for col in df.columns if col not in ['Modal_Price', 'Commodity']]
# X = df[feature_columns]
# y = df['Modal_Price']

# # Train the model
# lr_model = LinearRegression()
# lr_model.fit(X, y)

# # Request model
# class CommodityRequest(BaseModel):
#     commodity: str

# @app.post("/predict")
# def predict_price(request: CommodityRequest):
#     # Normalize user input
#     commodity_input = request.commodity.strip().title()
    
#     # Debug: Print available commodities
#     print(f"User input: '{commodity_input}'")
#     print("Available commodities:", list(df['Commodity'].unique()))

#     # Check if commodity exists
#     if commodity_input not in df['Commodity'].values:
#         return {"error": f"Commodity '{commodity_input}' not found. Check spelling!"}

#     # Transform commodity input
#     commodity_code = label_encoders['Commodity'].transform([commodity_input])[0]
    
#     # Debug: Print encoded commodity values
#     print(f"Encoded value for '{commodity_input}': {commodity_code}")
#     print("Unique encoded commodities:", df['Commodity_Encoded'].unique())

#     # Filter dataset for selected commodity
#     commodity_data = df[df['Commodity_Encoded'] == commodity_code]

#     # Debug: Print number of rows found
#     print(f"Rows found for '{commodity_input}': {len(commodity_data)}")

#     if commodity_data.empty:
#         return {"error": f"No data available for commodity '{commodity_input}'."}

#     # Get the latest entry
#     latest_entry = commodity_data.iloc[-1].copy()

#     # Ensure required features exist
#     if 'Day_Number' not in latest_entry:
#         return {"error": "Feature 'Day_Number' missing in dataset."}

#     # Forecast price intervals
#     future_days = [0, 3, 7, 15]
#     forecast_labels = ["Today", "In 3 days", "Next week", "Next 15 days"]
#     forecasted_prices = {}

#     for days, label in zip(future_days, forecast_labels):
#         latest_entry['Day_Number'] += days  # Adjust date feature
        
#         # Debug: Print features before prediction
#         print(f"Predicting for {label}: {latest_entry[feature_columns].values}")
        
#         future_price = lr_model.predict([latest_entry[feature_columns]])[0]
#         forecasted_prices[label] = round(future_price, 2)

#     return {
#         "current_price": forecasted_prices["Today"],
#         "forecast": forecasted_prices
#     }

