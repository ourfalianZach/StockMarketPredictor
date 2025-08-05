import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np


# trains model on training set and predicts on test set
def predict(train, test, predictors, model):
    model.fit(
        train[predictors], train["Target"]
    )  # train model on training set using predictors
    preds = model.predict_proba(test[predictors])[
        :, 1
    ]  # uses trained model to predict probabilities that the market goes up
    preds[preds >= 0.65] = 1  # if probability is greater than 0.6, set to 1
    preds[preds < 0.65] = 0  # if probability is less than 0.6, set to 0
    preds = pd.Series(
        preds, index=test.index, name="Predictions"
    )  # converts predictions to a series with the same index as the test set
    combined = pd.concat(
        [test["Target"], preds], axis=1
    )  # concatenates the target and predictions
    return combined


# trains model on past data and tests on future data
# start = starting index of the test set, 10 years of data (250 trading days per year), train first model with this data
# step = step size, 1 year of data, test the model on this data
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []  # list of dataframes to store all predictions, each dataframe is a year of predictions
    for i in range(
        start, data.shape[0], step
    ):  # loops through the data in steps of step size
        # training set (all data up to the current index)
        train = data.iloc[0:i].copy()
        # test set (current index to current index + step)
        test = data.iloc[i : (i + step)].copy()
        predictions = predict(
            train, test, predictors, model
        )  # predicts on the test set
        all_predictions.append(predictions)  # adds the predictions to the list
    return pd.concat(all_predictions)


# Function to compute RSI
def compute_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(gain, index=data.index).rolling(window).mean()
    roll_down = pd.Series(loss, index=data.index).rolling(window).mean()

    rs = roll_up / roll_down.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Function to compute MACD
def compute_macd(data, span1=12, span2=26, signal_span=9):
    ema1 = data["Close"].ewm(span=span1, adjust=False).mean()
    ema2 = data["Close"].ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal


sp500 = yf.Ticker("^GSPC")  # gets the data for the S&P 500
sp500 = sp500.history(period="max")
del sp500["Dividends"]  # deletes the dividends column
del sp500["Stock Splits"]  # deletes the stock splits column
sp500["Tomorrow"] = sp500["Close"].shift(-1)  # collumn with tomorrow's close price
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
# collumn with 1 if tomorrow's close price is greater than today's close price, 0 otherwise
sp500 = sp500.loc["1990-01-01":].copy()  # keeps only data from 1990-01-01 onwards


horizons = [2, 5, 60, 250, 1000]  # where we want to look at rolling averages
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(
        horizon
    ).mean()  # rolling average of the close price over the horizon
    ratio_column = f"Close_Ratio_{horizon}"  # name of the new column
    sp500[ratio_column] = (
        sp500["Close"] / rolling_averages["Close"]
    )  # close price divided by rolling average

    trend_column = f"Trend_{horizon}"  # name of the new column
    sp500[trend_column] = (
        sp500.shift(1).rolling(horizon).sum()["Target"]
    )  # sum of the number of days the market went up over the horizon
    new_predictors.append(ratio_column)
    new_predictors.append(trend_column)  # adds the new predictors to the list


sp500["RSI"] = compute_rsi(sp500)
sp500 = sp500.dropna(subset=["RSI"])
sp500["RSI_Overbought"] = (sp500["RSI"] > 70).astype(int)
sp500["RSI_Oversold"] = (sp500["RSI"] < 30).astype(int)
new_predictors += ["RSI", "RSI_Overbought", "RSI_Oversold"]

sp500["MACD"], sp500["MACD_Signal"] = compute_macd(sp500)
sp500["MACD_Bullish"] = (sp500["MACD"] > sp500["MACD_Signal"]).astype(int)
sp500["MACD_Bearish"] = (sp500["MACD"] < sp500["MACD_Signal"]).astype(int)
new_predictors += ["MACD_Bullish", "MACD_Bearish"]


# Compute Bollinger Bands (volatility)
sp500["20d_MA"] = sp500["Close"].rolling(20).mean()
sp500["20d_STD"] = sp500["Close"].rolling(20).std()
sp500["Upper_Band"] = sp500["20d_MA"] + (2 * sp500["20d_STD"])
sp500["Lower_Band"] = sp500["20d_MA"] - (2 * sp500["20d_STD"])
sp500["BB_Width"] = (sp500["Upper_Band"] - sp500["Lower_Band"]) / sp500["20d_MA"]
new_predictors += ["Upper_Band", "Lower_Band", "BB_Width"]

# Compute Volume Change
sp500["Volume_Change"] = sp500["Volume"].pct_change()
new_predictors.append("Volume_Change")

# Drop NaN values caused by rolling computations
sp500.replace([np.inf, -np.inf], np.nan, inplace=True)
sp500 = sp500.dropna()


# creates a random forest classifier model
# n_estimators = number of trees in the forest
# min_samples_split = minimum number of samples required to split an internal node
# random_state = random seed for reproducibility

model = RandomForestClassifier(
    n_estimators=500, min_samples_split=10, max_features="sqrt", random_state=1
)

predictions = backtest(sp500, model, new_predictors)  # backtests the model on the data

print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
