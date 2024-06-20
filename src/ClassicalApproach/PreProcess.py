import pandas as pd

input_file = "../../data/Tick/BTCGBP.csv"
output_file = 'processed_BTCGBP.csv'
rows_output_file = 'interval_processed_BTCGBP.csv'
rows_diff_output_file = 'interval_diff_processed_BTCGBP.csv'


# moving average preprocessing (no longer used in the classical approach)
def moving_average_processing():
    df = pd.read_csv(input_file, delimiter='|', header=None)
    df.columns = ['Index', 'Price', 'Volume', 'Time', 'Type']
    df['MA100'] = df['Price'].rolling(window=100, min_periods=1).mean()

    df.to_csv(output_file, index=False)

    print(f"Filtered data has been saved to {output_file}")


# optimise data set size by filtering only rows that make the grid level change movement
def grid_interval_filter():
    df = pd.read_csv(output_file)

    last_index = None
    last_price = None
    threshold_up = None
    threshold_down = None
    filtered_rows = []
    grid_interval = 500

    for index, row in df.iterrows():
        if last_index is None:
            last_index = index
            last_price = row['Price']
            threshold_up = last_price + grid_interval
            threshold_down = last_price - grid_interval
            filtered_rows.append(index)
        else:
            if row['Price'] >= threshold_up or row['Price'] <= threshold_down:
                last_index = index
                last_price = row['Price']
                threshold_up = last_price + grid_interval
                threshold_down = last_price - grid_interval
                filtered_rows.append(index)

    filtered_df = df.loc[filtered_rows]

    filtered_df.to_csv(rows_output_file, index=False)


# strength metric for normalisation
def moving_average_strength():
    df = pd.read_csv(rows_output_file)

    df['MADiff'] = df['Price'] - df['MA100']
    df.to_csv(rows_diff_output_file, index=False)

moving_average_strength()
