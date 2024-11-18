import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from pandas import DataFrame, concat, Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- Fungsi Pendukung ----

# Mengubah urutan waktu menjadi data supervised
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Mendiferensiasi data untuk membuat stasioner
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# Inversi diferensiasi
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Scaling data ke rentang [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# Membalik scaling data
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row).reshape(1, len(new_row))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# Melatih model LSTM
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshaping the input to (samples, timesteps, features)
    
    # Define n_steps and n_features
    n_steps = X.shape[1]  # This is 1 since each sample has 1 timestep
    n_features = X.shape[2]  # Number of features in each timestep
    
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(n_steps, n_features)))  # input_shape based on X shape
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        # No need to manually reset states; TensorFlow will manage it automatically
    return model


# Memprediksi nilai
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

# Multi-step prediksi
def forecast_lstm_multi_steps(model, batch_size, X, n_steps, raw_values, scaler):
    predictions = []
    input_data = X
    for i in range(n_steps):
        yhat = model.predict(input_data.reshape(1, 1, len(input_data)), batch_size=batch_size)[0, 0]
        yhat_inverted = invert_scale(scaler, input_data, yhat)
        yhat_inverted = inverse_difference(raw_values, yhat_inverted, len(raw_values) - len(input_data) + i)
        predictions.append(yhat_inverted)
        input_data = np.append(input_data[1:], yhat)
    return predictions

# ---- Aplikasi Streamlit ----
st.set_page_config(page_title="Aplikasi Peramalan PM10", layout="wide")

# Global: File Upload
uploaded_file = st.sidebar.file_uploader("Unggah file dataset (.csv atau .xlsx)", type=["csv", "xlsx"])

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Dataset", "Peramalan"])

# Halaman Beranda
if page == "Beranda":
    st.title("Aplikasi Peramalan PM10 dengan LSTM")
    st.write("""
    Aplikasi ini digunakan untuk memprediksi konsentrasi PM10 menggunakan model Long Short-Term Memory (LSTM).
    Anda dapat mengunggah dataset PM10, melihat data historis, dan melakukan prediksi konsentrasi untuk hari-hari mendatang.
    Navigasikan melalui sidebar untuk memulai.
    """)
    st.image("https://via.placeholder.com/600x300.png?text=Ilustrasi+Prediksi", use_container_width=True)

# Halaman Dataset
elif page == "Dataset":
    st.title("Dataset PM10")
    if uploaded_file:
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(data.head())
        data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
        plt.figure(figsize=(10, 5))
        plt.plot(data['Tanggal'], data['PM10'], label="Konsentrasi PM10")
        plt.xlabel("Tanggal")
        plt.ylabel("PM10")
        plt.title("Konsentrasi PM10")
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("Silakan unggah file dataset untuk melihat data.")

# Halaman Peramalan
elif page == "Peramalan":
    st.title("Peramalan PM10")
    if uploaded_file:
        days = st.number_input("Jumlah hari untuk prediksi", min_value=1, max_value=300)
        if st.button("Prediksi"):
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            data = data.sort_values('Tanggal')
            raw_values = data['PM10'].values
            diff_values = difference(raw_values, 1)
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            split_index = int(len(supervised_values) * 0.8)
            train, test = supervised_values[:split_index], supervised_values[split_index:]
            scaler, train_scaled, test_scaled = scale(train, test)
            lstm_model = fit_lstm(train_scaled, 1, 20, 7)
            X = test_scaled[-1, 0:-1]
            predictions = forecast_lstm_multi_steps(lstm_model, 1, X, days, raw_values, scaler)
            pred_dates = pd.date_range(data['Tanggal'].iloc[-1], periods=days + 1)[1:]
            results = pd.DataFrame({"Tanggal": pred_dates, "Prediksi": np.exp(predictions)})
            st.dataframe(results)
            plt.figure(figsize=(10, 5))
            plt.plot(pred_dates, np.exp(predictions), label="Prediksi PM10")
            plt.xlabel("Tanggal")
            plt.ylabel("PM10")
            plt.title("Prediksi Konsentrasi PM10")
            plt.legend()
            st.pyplot(plt)
    else:
        st.write("Silakan unggah file dataset di halaman Dataset terlebih dahulu.")
