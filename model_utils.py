import joblib  # used to save and load the model
import os
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_PATH = "rf_model.pkl"  # path to the model


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# moving average convergence divergence
# difference between short-term and long-term exponential moving averages
def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


# show volatility based on 20 day average and 2 standard deviations
def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


# percentage change in volume
def compute_volume_change(series):
    return series.pct_change()


# ratio of close to open price
def compute_close_ratios(df):
    open_close_ratio = df["Close"] / df["Open"]
    high_low_ratio = df["High"] / df["Low"]
    return open_close_ratio, high_low_ratio


# difference between current price and price 5 days ago
def compute_trends(series, window=5):
    return series.diff(window)


# calls all indicators and adds them to the dataframe
def prepare_features(df):
    df["RSI"] = compute_rsi(df["Close"])
    macd, signal = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    upper_band, lower_band = compute_bollinger_bands(df["Close"])
    df["Bollinger_Upper"] = upper_band
    df["Bollinger_Lower"] = lower_band
    df["Volume_Change"] = compute_volume_change(df["Volume"])
    open_close_ratio, high_low_ratio = compute_close_ratios(df)
    df["Open_Close_Ratio"] = open_close_ratio
    df["High_Low_Ratio"] = high_low_ratio
    df["Trend_5"] = compute_trends(df["Close"], 5)
    df["Trend_10"] = compute_trends(df["Close"], 10)
    df = df.dropna()
    return df


# creates a target column that is 1 if the next day's close is higher than the current day's close, 0 otherwise
def create_target(df):
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    return df


# trains the model and saves it to the model path
def train_model(df):
    features = [
        "RSI",
        "MACD",
        "MACD_Signal",
        "Bollinger_Upper",
        "Bollinger_Lower",
        "Volume_Change",
        "Open_Close_Ratio",
        "High_Low_Ratio",
        "Trend_5",
        "Trend_10",
    ]
    X = df[features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(
        n_estimators=500, min_samples_split=10, max_features="sqrt", random_state=1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        df = yf.download("^GSPC", start="2010-01-01", interval="1d")
        df = prepare_features(df)
        df = create_target(df)
        model = train_model(df)
    else:
        model = joblib.load(MODEL_PATH)
    return model


def predict_sp500():
    df = yf.download("^GSPC", start="2010-01-01", interval="1d")
    df = prepare_features(df)
    df = create_target(df)
    model = load_model()
    latest_features = df.iloc[-1][
        [
            "RSI",
            "MACD",
            "MACD_Signal",
            "Bollinger_Upper",
            "Bollinger_Lower",
            "Volume_Change",
            "Open_Close_Ratio",
            "High_Low_Ratio",
            "Trend_5",
            "Trend_10",
        ]
    ].values.reshape(1, -1)
    proba = model.predict_proba(latest_features)[0][1]
    prediction = "UP" if proba >= 0.6 else "DOWN"
    return {"prediction": prediction, "probability": proba}
