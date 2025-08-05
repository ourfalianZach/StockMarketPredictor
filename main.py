import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd


# trains model on training set and predicts on test set
def predict(train, test, predictors, model):
    model.fit(
        train[predictors], train["Target"]
    )  # train model on training set using predictors
    preds = model.predict_proba(test[predictors])[
        :, 1
    ]  # uses trained model to predict probabilities that the market goes up
    preds[preds >= 0.6] = 1  # if probability is greater than 0.6, set to 1
    preds[preds < 0.6] = 0  # if probability is less than 0.6, set to 0
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

sp500 = sp500.dropna()  # drop any rows with missing values

# creates a random forest classifier model
# n_estimators = number of trees in the forest
# min_samples_split = minimum number of samples required to split an internal node
# random_state = random seed for reproducibility
model = RandomForestClassifier(n_estimators=200, min_samples_split=25, random_state=1)

predictions = backtest(sp500, model, new_predictors)  # backtests the model on the data

print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
