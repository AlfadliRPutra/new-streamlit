import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from pandas import DataFrame, Series, concat
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math

# Functions for RMSE and MAPE
def calculate_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted) * 100

# Time series to supervised learning
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Functions for differencing
def difference(dataset, interval=1):
    diff = [dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))]
    return Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Scale data
def scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    return scaler, train_scaled

# Inverse scaling
def invert_scale(scaler, X, value):
    new_row = np.array([x for x in X] + [value])
    new_row = new_row.reshape(1, -1)
    inverted = scaler.inverse_transform(new_row)
    return inverted[0, -1]

# Fit LSTM model
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
    return model

# Forecast using LSTM
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

# Convert dimensions
def convertDimension(value):
    return np.reshape(value, (value.shape[0], 1, value.shape[0]))

# File path for model
model_file_path = 'lstm_model.pkl'

# Save and load model functions
def save_model(model, model_file_path):
    joblib.dump(model, model_file_path)

def load_model(model_file_path):
    if os.path.exists(model_file_path):
        return joblib.load(model_file_path)
    else:
        return None

# Streamlit app
st.title("Aplikasi Peramalan")

# File uploader
uploaded_file = st.sidebar.file_uploader("Unggah file CSV atau Excel Anda", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    if not st.session_state.model_trained:
        with st.spinner("Memproses dan melatih model..."):
            # Load data
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'csv':
                series = pd.read_csv(uploaded_file, usecols=[0, 1], engine='python', header=0)
            elif file_extension in ['xlsx', 'xls']:
                series = pd.read_excel(uploaded_file, usecols=[0, 1], header=0)

            # Process data
            series['Tanggal'] = pd.to_datetime(series['Tanggal'], format='%d/%m/%Y')
            series.set_index('Tanggal', inplace=True)

            raw_values = series['PM10'].values.reshape(-1, 1)
            diff_values = difference(raw_values, 1)
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            train = supervised_values[0:]
            scaler, train_scaled = scale(train)

            # Train model
            lstm_model = fit_lstm(train_scaled, 1, 100, 5)
            save_model(lstm_model, model_file_path)

            # Update session state
            st.session_state.data = series
            st.session_state.scaler = scaler
            st.session_state.model_trained = True

    st.sidebar.header("Navigasi")
    selection = st.sidebar.radio("Pilih Tampilan", ["Dataset", "Peramalan"])

    if selection == "Dataset":
        st.subheader("Tabel")
        st.write(st.session_state.data)
    
        st.subheader("Visualisasi data")
        plt.figure(figsize=(10, 6))
        plt.plot(st.session_state.data.index, st.session_state.data['PM10'], label="Konsentrasi PM10", color="blue")
        plt.xlabel("Tanggal")
        plt.ylabel("Konsentrasi PM10")
        plt.title("Visualisasi Konsentrasi PM10 dari Dataset")
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)

    if selection == "Peramalan":
        st.subheader("Peramalan")
        
        if st.session_state.model_trained:
            lstm_model = load_model(model_file_path)
            raw_values = st.session_state.data['PM10'].values.reshape(-1, 1)
            diff_values = difference(raw_values, 1)
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            
            # Split data
            split_index = int(0.8 * len(supervised_values))
            train, test = supervised_values[:split_index], supervised_values[split_index:]
            train_scaled = st.session_state.scaler.transform(train)
            test_scaled = st.session_state.scaler.transform(test)

            # Make predictions on test data
            predictions = []
            for i in range(len(test_scaled)):
                X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
                yhat = forecast_lstm(lstm_model, 1, X)
                yhat_inverted = invert_scale(st.session_state.scaler, X, yhat)
                yhat_inverted = inverse_difference(raw_values, yhat_inverted, len(test_scaled) + 1 - i)
                predictions.append(yhat_inverted)

            # Calculate RMSE and MAPE
            actual_test_values = raw_values[split_index + 1:].flatten()
            rmse = calculate_rmse(actual_test_values, predictions)
            mape = calculate_mape(actual_test_values, predictions)

            # Display metrics
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")
            st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
            
            # DataFrame for actual vs predicted
            test_dates = st.session_state.data.index[split_index + 1:]
            result_df = pd.DataFrame({
                'Tanggal': test_dates,
                'Aktual': actual_test_values,
                'Prediksi': np.round(np.array(predictions).flatten())  # Ensure predictions are flattened
            }).set_index('Tanggal')
    
            st.subheader("Tabel Data Aktual vs Prediksi")
            st.dataframe(result_df)
    
            # Plot predictions vs actual
            plt.figure(figsize=(15, 7))
            plt.plot(test_dates, actual_test_values, label="Aktual PM10")
            plt.plot(test_dates, predictions, label="Prediksi PM10", linestyle="--", color="red")
            plt.xlabel("Tanggal")
            plt.ylabel("Konsentrasi PM10")
            plt.title("Hasil Prediksi vs Aktual pada Data Testing")
            plt.legend()
            st.pyplot(plt)

            st.subheader("Peramalan")
            # Input for future predictions
            future_days = st.number_input("Pilih jumlah hari untuk diprediksi:", min_value=0, max_value=300)

            if future_days > 0:
                st.subheader(f"Peramalan untuk {future_days} hari ke depan")
                lastPredict = train_scaled[-1, 0].reshape(1, 1, 1)
                future_predictions = []

                for _ in range(future_days):
                    yhat = forecast_lstm(lstm_model, 1, lastPredict)
                    future_predictions.append(yhat)
                    lastPredict = convertDimension(np.array([[yhat]]))

                # Invert scaling and differencing for future predictions
                future_predictions_inverted = []
                for i in range(len(future_predictions)):
                    tmp_result = invert_scale(st.session_state.scaler, [0], future_predictions[i])
                    tmp_result = inverse_difference(raw_values, tmp_result, i + 1)
                    future_predictions_inverted.append(tmp_result)

                # Flatten future_predictions_inverted before creating the DataFrame
                future_predictions_inverted = np.array(future_predictions_inverted).flatten()
                
                # Create DataFrame for future predictions
                last_date = st.session_state.data.index[-1]
                future_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=future_days, freq='D')
                
                future_df = pd.DataFrame({
                    'Tanggal': future_index,
                    'Prediksi': np.round(future_predictions_inverted)  # Ensure predictions are rounded and flattened
                }).set_index('Tanggal')


                # Display future predictions table
                st.subheader("Tabel Prediksi")
                st.dataframe(future_df)

                # Plot future predictions
                plt.figure(figsize=(15, 7))
                plt.plot(st.session_state.data.index, st.session_state.data['PM10'], label="Data Asli PM10")
                plt.plot(future_df.index, future_predictions_inverted, label="Prediksi LSTM", linestyle="--", color="red")
                plt.axvline(x=last_date, color='blue', linestyle='--', label="Batas Data Asli")
                plt.xlabel("Tanggal")
                plt.ylabel("Konsentrasi PM10")
                plt.title(f"Konsentrasi PM10 dan Prediksi LSTM untuk {future_days} Hari ke Depan")
                plt.legend()
                st.pyplot(plt)
