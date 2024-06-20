import pandas as pd
import numpy as np

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    df['date'] = pd.to_datetime(df['date'], unit='ms')

    df.set_index('date', inplace=True)

    # resample the data to 24-hour periods and process the price and volume data:
    resampled = df.resample('24h').agg({
        'price': ['max', 'min', 'mean', 'std', 'first', 'last'],
        'volume': 'sum'
    })

    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]

    resampled['price_change'] = resampled['price_last'] - resampled['price_first']

    resampled.drop(columns=['price_first', 'price_last'], inplace=True)

    resampled.rename(columns={
        'price_max': 'highest_price',
        'price_min': 'lowest_price',
        'price_mean': 'average_price',
        'price_std': 'price_std_dev',
        'volume_sum': 'volume_sum'
    }, inplace=True) # formatting

    resampled.reset_index(inplace=True)
    resampled.to_csv(output_file, index=False)

input_file = 'BTCGBP.csv'
output_file = 'preprocessed_BTCGBP.csv'

process_csv(input_file, output_file)