import yfinance as yf

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500["Tomorrow"] = sp500["Close"].shift(-1)  # collumn with tomorrow's close price
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
# collumn with 1 if tomorrow's close price is greater than today's close price, 0 otherwise
sp500 = sp500.loc["1990-01-01":].copy()  # keeps only data from 1990-01-01 onwards
print(sp500)
