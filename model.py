import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyodbc


# Load the data from your SQL Server DB
def load_data(query):
    # Replace the following connection parameters with your own SQL Server credentials
    server = 'your_server_name'
    database = 'your_database_name'
    username = 'your_username'
    password = 'your_password'
    driver = '{ODBC Driver 17 for SQL Server}'  # Use the appropriate driver for your SQL Server version

    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

    with pyodbc.connect(connection_string) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def is_heavy_rain(rain_value, station_id):
    # Replace with the function that distinguishes an amount of rain is heavy for a station
    pass


def get_heavy_rain_data():
    # Load data
    rainfall_data = load_data("SELECT * FROM your_rainfall_data_table_name")
    flood_data = load_data("SELECT * FROM your_flood_data_table_name")
    stations_info = load_data("SELECT * FROM your_stations_info_table_name")

    # Merge data on Station_Id and Date
    merged_data = pd.merge(rainfall_data, flood_data, on=['Station_Id', 'Date'], how='outer')
    merged_data = pd.merge(merged_data, stations_info, on='Station_Id', how='left')

    # Filter the data for heavy rain
    merged_data['is_heavy_rain'] = merged_data.apply(lambda row: is_heavy_rain(row['value'], row['Station_Id']), axis=1)
    heavy_rain_data = merged_data[merged_data['is_heavy_rain']]

    return heavy_rain_data


def train_model():
    heavy_rain_data = get_heavy_rain_data()

    # Prepare the dataset
    X = heavy_rain_data[['Station_Id', 'Lat', 'Long', 'value']]
    y = heavy_rain_data['discharge_value']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    return rf_model, scaler


def predict_flood_probabilities(heavy_rain_data):
    rf_model, scaler = train_model()
    X_pred = heavy_rain_data[['Station_Id', 'Lat', 'Long', 'value']]
    X_pred_scaled = scaler.transform(X_pred)

    # Predict flood probabilities
    flood_probabilities = rf_model.predict(X_pred_scaled)
    heavy_rain_data['flood_probability'] = flood_probabilities

    return heavy_rain_data
