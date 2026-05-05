import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

st.title("📈 AI Stock Scanner System")

# -------------------------------
# MODULE 1: DATA FETCH
# -------------------------------
def fetch_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    return data

# -------------------------------
# MODULE 2: INDICATORS
# -------------------------------
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def moving_average(data, window=20):
    return data['Close'].rolling(window).mean()

# -------------------------------
# MODULE 3: SCANNER
# -------------------------------
def scan_stock(data):
    data['RSI'] = compute_rsi(data)
    latest = data.iloc[-1]

    if latest['RSI'] < 30:
        return "BUY (Oversold)"
    elif latest['RSI'] > 70:
        return "SELL (Overbought)"
    else:
        return "HOLD"

# -------------------------------
# MODULE 4: ML MODEL
# -------------------------------
def train_model(data):
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)

    X = data[['Open','High','Low','Close','Volume']]
    y = data['Target']

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# -------------------------------
# MODULE 5: SENTIMENT
# -------------------------------
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# -------------------------------
# UI
# -------------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Scan"):
    data = fetch_data(ticker)

    if data.empty:
        st.error("Invalid ticker!")
    else:
        signal = scan_stock(data)
        model = train_model(data)

        prediction = model.predict([data[['Open','High','Low','Close','Volume']].iloc[-1]])[0]

        sentiment = analyze_sentiment("Stock market is bullish today")

        st.subheader(f"Signal: {signal}")
        st.subheader(f"AI Prediction: {'UP' if prediction==1 else 'DOWN'}")
        st.subheader(f"Sentiment Score: {sentiment}")

        st.line_chart(data['Close'])
