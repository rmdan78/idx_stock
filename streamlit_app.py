import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px


pipeline = joblib.load('model.pkl')


def calculate_features(data):
    data['MA_3'] = data['Close'].rolling(window=3).mean().shift(1)
    data['MA_5'] = data['Close'].rolling(window=5).mean().shift(1)
    data['MA_10'] = data['Close'].rolling(window=10).mean().shift(1)
    data['Return'] = data['Close'].pct_change().shift(1)
    data['Volatility'] = data['Return'].rolling(window=10).std().shift(1)
    data = data.fillna(0)
    return data


st.title('Stock Price Prediction')
st.info("This app uses a Neural Network-based model for stock price prediction. The model has been trained on historical stock data and includes technical features.")
st.header('Input Ticker Stock')
ticker = st.text_input('Ticker', value='BBCA.JK')


if st.button('Predict'):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=360)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if not stock_data.empty:
        stock_data = calculate_features(stock_data)
        
        latest_data = stock_data.iloc[-1]
        Open = latest_data['Open']
        High = latest_data['High']
        Low = latest_data['Low']
        Close = latest_data['Close']
        Volume = latest_data['Volume']

        input_data = pd.DataFrame({
            'Open': [Open],
            'High': [High],
            'Low': [Low],
            'Close': [Close],
            'Volume': [Volume],
            'MA_3': [latest_data['MA_3']],
            'MA_5': [latest_data['MA_5']],
            'MA_10': [latest_data['MA_10']],
            'Return': [latest_data['Return']],
            'Volatility': [latest_data['Volatility']]
        })
        
        prediction = pipeline.predict(input_data)
        predicted_price = float(prediction[0]) 
        fig = px.line(
        stock_data,
        x=stock_data.index,
        y=stock_data['Close'].squeeze(), 
        title="Harga Penutupan Saham 3 Bulan Terakhir"
        )

        st.plotly_chart(fig)
        Close = float(latest_data['Close'])

        st.subheader(f'Latest Close : Rp{Close:,.2f}')
        st.subheader(f'Predicted Close: Rp{predicted_price:,.2f}')
        st.warning("⚠️ This app is for educational purposes only. The stock price predictions provided here are not financial advice. Please do your own research before making any investment decisions.")


    else:
        st.error(f'No data found for ticker: {ticker}')

