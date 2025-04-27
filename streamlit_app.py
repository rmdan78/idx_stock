import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import pickle
import os

# Cek file yang ada di direktori
st.write('Files in current directory:', os.listdir())

# Muat model menggunakan pickle
model_filename = 'stock_price_pipeline_updated.pkl'
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        pipeline = pickle.load(f)
else:
    st.error(f"Model file {model_filename} tidak ditemukan.")
    st.stop()

# Fungsi untuk menghitung fitur tambahan
def calculate_features(data):
    data['MA_3'] = data['Close'].rolling(window=3).mean().shift(1)
    data['MA_5'] = data['Close'].rolling(window=5).mean().shift(1)
    data['MA_10'] = data['Close'].rolling(window=10).mean().shift(1)
    data['Return'] = data['Close'].pct_change().shift(1)
    data['Volatility'] = data['Return'].rolling(window=10).std().shift(1)
    data = data.fillna(0)
    return data

# Judul aplikasi
st.title('📈 Stock Price Prediction')

# Input ticker saham
st.header('Input Ticker Saham')
ticker = st.text_input('Ticker', value='AAPL')

# Tombol untuk mengambil data dan melakukan prediksi
if st.button('Predict'):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=360)

    # Tambahkan loading spinner agar UX lebih bagus
    with st.spinner('Mengambil data dari Yahoo Finance...'):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if not stock_data.empty:
        stock_data = calculate_features(stock_data)
        latest_data = stock_data.iloc[-1]
        
        # Membuat DataFrame untuk prediksi
        input_data = pd.DataFrame({
            'Open': [latest_data['Open']],
            'High': [latest_data['High']],
            'Low': [latest_data['Low']],
            'Close': [latest_data['Close']],
            'Volume': [latest_data['Volume']],
            'MA_3': [latest_data['MA_3']],
            'MA_5': [latest_data['MA_5']],
            'MA_10': [latest_data['MA_10']],
            'Return': [latest_data['Return']],
            'Volatility': [latest_data['Volatility']]
        })

        # Prediksi
        prediction = pipeline.predict(input_data)

        # Tampilkan grafik harga
        fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'1 Year Close Prices for {ticker}')
        st.plotly_chart(fig)

        # Tampilkan harga asli dan prediksi
        st.subheader(f'Latest Close Price for {ticker}: Rp{latest_data["Close"]:.2f}')
        st.subheader(f'Predicted Close Price for {ticker}: Rp{prediction[0]:.2f}')
    else:
        st.error(f'Data untuk ticker {ticker} tidak ditemukan atau kosong. Pastikan ticker valid dan koneksi internet aktif.')
