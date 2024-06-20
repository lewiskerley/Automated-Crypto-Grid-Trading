import pandas as pd
import numpy as np

# WMA and STO processing


def calculate_wma(input_file, output_file):
    df = pd.read_csv(input_file, header=0)

    # WMA
    step = 1 # only evaluate every n ticks
    n = 20000 # About 24 hours = 20,000
    length = df.shape[0]
    def calculate_wma(x):
        nonlocal i
        if i % 10000 == 0:
          print(f"Progress: {i} / {round(length / step)} - {round((i / round(length / step)) * 1000) / 10}%")  # Print progress
        i += 1
        return (x * range(n, 0, -1)).sum() / (n * (n + 1) / 2)
    i = 1
    wma = df['price'].rolling(window=n, step=step).apply(calculate_wma, raw=True)
    df['wma'] = wma
    df.ffill(inplace=True)
    df.dropna(inplace=True) # The first step* are dropped as they are incalculable.

    # ensure all data is float
    df = pd.DataFrame(df.values.astype('float32'))
    df.columns = ['date', 'tick_id', 'price', 'volume', 'type', 'wma']

    df.to_csv(output_file, index=False)

# calculate_wma("preprocessed_BTCGBP.csv", "preprocessed_BTCGBP_wma.csv")

# df = pd.read_csv("preprocessed_BTCGBP.csv", header=0)
# df2 = pd.read_csv("preprocessed_BTCGBP_wma.csv", header=0)
# df2['date'] = df.shift(-19999)['date']
# df2.to_csv("preprocessed_BTCGBP_wma.csv", index=False)




def calculate_wma(input_file, output_file):
    # read the CSV file
    df = pd.read_csv(input_file, header=0)
    df['price_twenty_ahead'] = df.shift(-20)['price']
    df.dropna(inplace=True) # The last 20 are dropped as they are incalculable.

    # ensure all data is float
    df = pd.DataFrame(df.values.astype('float32'))
    df.columns = ['date', 'tick_id', 'price', 'volume', 'type', 'wma', 'price_twenty_ahead']

    # write the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# calculate_wma("preprocessed_BTCGBP_wma.csv", "preprocessed_BTCGBP_wma_twenty.csv")