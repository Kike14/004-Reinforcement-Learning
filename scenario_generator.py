import tensorflow as tf
import numpy as np
import pandas as pd
from ta import add_all_ta_features

def load_generator_model(path="generador.keras"):
    return tf.keras.models.load_model(path)

def generate_scenarios(generator, precios_df, num_scenarios=1000, scenario_length=252):
    scenarios = []
    noise = tf.random.normal([num_scenarios, scenario_length, 1])
    generated_returns = generator(noise, training=False).numpy()

    price_scenarios = []
    for returns in generated_returns:
        initial_price = precios_df['Adj Close'].sample(n=1).iloc[0]
        prices = [initial_price]
        for r in returns:
            prices.append(prices[-1] * np.exp(r))

        df = pd.DataFrame({'Close': prices})
        df['High'] = df['Close'] * (1 + np.random.uniform(0.001, 0.01, size=len(df)))
        df['Low'] = df['Close'] * (1 - np.random.uniform(0.001, 0.01, size=len(df)))
        df['Open'] = df['Close'].shift(1).fillna(df['Close'][0])
        df['Volume'] = np.random.uniform(1000, 5000, size=len(df))
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        price_scenarios.append(df)

    return price_scenarios