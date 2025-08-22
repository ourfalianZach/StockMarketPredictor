S&P 500 Market Predictor

This project is a full-stack web application that predicts whether the S&P 500 index will go up 
or down on the next trading day using machine learning. It combines a trained Random Forest Classifier with a 
Flask REST API backend and a React.js frontend for interactive display of predictions.


Features:

-Machine learning model trained on historical S&P 500 data
-Technical indicators used: RSI, MACD, Bollinger Bands, volume change, price ratios, and trend features
-Real-time data fetching from Yahoo Finance using yfinance
-Flask REST API that returns prediction results via a /predict endpoint
-React.js frontend that fetches and displays predictions
-Model is automatically retrained and saved using joblib if not already stored

Tech Stack:

-Frontend: React.js, HTML, CSS
-Backend: Python, Flask
-Machine Learning: scikit-learn, pandas, numpy, joblib
-Data: yfinance (Yahoo Finance API)
-Tools: Git, npm, VS Code

Model Details:

-Algorithm: RandomForestClassifier from scikit-learn
-Training Data: Daily S&P 500 historical data starting from 2010
-Target: Binary classification of whether the next day's closing price is higher than today
-Evaluation: 80/20 train/test split, preserving time-series integrity (no shuffling)


Developed by Zachary Ourfalian
Data provided via Yahoo Finance
Machine learning implemented with scikit-learn and pandas