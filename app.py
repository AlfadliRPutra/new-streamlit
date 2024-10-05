import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from pandas import DataFrame, Series, concat
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math
import os
import joblib

# Helper functions

# Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=7):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# Invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Scale train and test data to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# Inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# Fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# Make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

# Convert list to one-dimensional array
def toOneDimension(value):
    return np.asarray(value)

# Convert to multi-dimensional array
def convertDimension(value):
    return np.reshape(value, (value.shape[0], 1, value.shape[0]))

# Root Mean Squared Error
def calculate_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

# Mean Absolute Percentage Error
def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted) * 100

# Streamlit app starts here
st.title("Aplikasi Peramalan PM10 dengan LSTM")

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

    # Load data
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'csv':
        series = pd.read_csv(uploaded_file, usecols=[0, 1], engine='python', header=0)
    elif file_extension in ['xlsx', 'xls']:
        series = pd.read_excel(uploaded_file, usecols=[0, 1], header=0)

    series['Tanggal'] = pd.to_datetime(series['Tanggal'], format='%d/%m/%Y')
    series.set_index('Tanggal', inplace=True)

    raw_values = series['PM10'].values.reshape(-1, 1)
    diff_values = difference(raw_values, 1)
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # Splitting the data into training and testing sets
    split_index = int(0.8 * len(supervised_values))
    train, test = supervised_values[:split_index], supervised_values[split_index:]
    scaler, train_scaled, test_scaled = scale(train, test)

    # Train the model
    if not st.session_state.model_trained:
        with st.spinner("Melatih model LSTM..."):
            lstm_model = fit_lstm(train_scaled, batch_size=1, nb_epoch=100, neurons=5)
            st.session_state.model = lstm_model
            st.session_state.scaler = scaler
            st.session_state.model_trained = True

    # Sidebar navigation
    st.sidebar.header("Navigasi")
    selection = st.sidebar.radio("Pilih Tampilan", ["Dataset", "Peramalan"])

    if selection == "Dataset":
        st.subheader("Tabel")
        st.write(series)

        st.subheader("Visualisasi Data PM10")
        plt.figure(figsize=(10, 6))
        plt.plot(series.index, series['PM10'], label="Konsentrasi PM10", color="blue")
        plt.xlabel("Tanggal")
        plt.ylabel("Konsentrasi PM10")
        plt.title("Visualisasi Konsentrasi PM10 dari Dataset")
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)

    if selection == "Peramalan":
        st.subheader("Peramalan PM10")

        # Predict future PM10 values
        future_days = st.number_input("Pilih jumlah hari untuk diprediksi:", min_value=1, max_value=300)

        if future_days > 0:
            st.subheader(f"Peramalan untuk {future_days} hari ke depan")

            lastPredict = train_scaled[-1, 0].reshape(1, 1, 1)
            future_predictions = []

            for _ in range(future_days):
                yhat = forecast_lstm(st.session_state.model, 1, lastPredict)
                future_predictions.append(yhat)
                lastPredict = convertDimension(np.array([[yhat]]))

            # Invert scaling and differencing
            future_predictions_inverted = []
            for i in range(len(future_predictions)):
                tmp_result = invert_scale(st.session_state.scaler, [0], future_predictions[i])
                tmp_result = inverse_difference(raw_values, tmp_result, i + 1)
                future_predictions_inverted.append(tmp_result)

            future_predictions_inverted = np.array(future_predictions_inverted).flatten()

            # Create a DataFrame for future predictions
            last_date = series.index[-1]
            future_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=future_days, freq='D')

            future_df = pd.DataFrame({
                'Tanggal': future_index,
                'Prediksi': np.round(future_predictions_inverted)
            }).set_index('Tanggal')

            st.subheader("Tabel Prediksi")
            st.dataframe(future_df)

            # Plot future predictions
            plt.figure(figsize=(15, 7))
            plt.plot(series.index, series['PM10'], label="Data Asli PM10")
            plt.plot(future_df.index, future_df['Prediksi'], label="Prediksi PM10", linestyle="--", color="red")
            plt.axvline(x=last_date, color='blue', linestyle='--', label="Batas Data Asli")
            plt.xlabel("Tanggal")
            plt.ylabel("Konsentrasi PM10")
            plt.title(f"Prediksi LSTM untuk {future_days} Hari ke Depan")
            plt.legend()
            st.pyplot(plt)
