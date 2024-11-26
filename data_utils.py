import pandas as pd
import yfinance as yf
from ta import add_all_ta_features

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    prices = data
    r = (np.log(prices[['Adj Close']] / prices[['Adj Close']].shift(1))).dropna()
    mean = r.mean()
    std = r.std()
    r_norm = (r - mean) / std
    return prices, r, r_norm, mean, std

def split_and_filter_by_year(data, columns):
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo datetime.")

    datasets_by_year = []
    for year in data.index.year.unique():
        yearly_data = data[data.index.year == year].copy()
        if not all(col in yearly_data.columns for col in columns):
            yearly_data = add_all_ta_features(
                yearly_data, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
        filtered_data = yearly_data[columns].reset_index(drop=True)
        datasets_by_year.append(filtered_data)

    return datasets_by_year