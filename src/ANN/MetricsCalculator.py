import pandas as pd
import numpy as np
import os

# mapping the saved algorithm results into standardised profit units
def map_profit(level_outcome):
    level_outcome = int(level_outcome)
    if level_outcome == 0 or abs(level_outcome) == 1:
        return 100
    elif abs(level_outcome) == 2:
        return 0
    elif abs(level_outcome) == 3:
        return -600
    else:
        raise ValueError(f"Unexpected LevelOutcome value: {level_outcome}")

def calculate_metrics(df):
    df = df[~df["LevelOutcome"].isin(["Pass", "Unknown"])]
    df['Profit'] = df['LevelOutcome'].apply(map_profit)
    
    df['CumulativeProfit'] = df['Profit'].cumsum()
    print(df['CumulativeProfit'])
    
    roi = ((df['CumulativeProfit'].iloc[-1] / (600 / 0.01)) * 100) + 1 # risking 1% of the account
    
    roll_max = df['CumulativeProfit'].cummax()
    daily_drawdown = df['CumulativeProfit'] - roll_max
    mdd = daily_drawdown.min()
    
    volatility = df['Profit'].std() / 100
    
    sharpe_ratio = ((roi / 100)) / volatility

    count = len(df['Profit'])
    
    return roi, mdd, volatility, sharpe_ratio, count # metrics displayed in the report

metrics = {
    'Interval': [],
    'ROI': [],
    'MDD': [],
    'Volatility': [],
    'Sharpe Ratio': [],
    'Count': []
}

file_path = "ann_results.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    roi, mdd, volatility, sharpe_ratio, count = calculate_metrics(df)
    
    # Store the results in the dictionary
    metrics['Interval'].append(500)
    metrics['ROI'].append(roi)
    metrics['MDD'].append(mdd)
    metrics['Volatility'].append(volatility)
    metrics['Sharpe Ratio'].append(sharpe_ratio)
    metrics['Count'].append(count)
else:
    print(f"File {file_path} does not exist.")

metrics_df = pd.DataFrame(metrics)

print(metrics_df)