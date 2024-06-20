import pandas as pd
import numpy as np
from matplotlib import pyplot
import math
import time

# Load the CSV data into a DataFrame
# data = pd.read_csv("../../data/Tick/BTCGBP.csv", header=None)
# print("Example data:\n", data.head(2))


def split_csv(input_file, output_file):
    df = pd.read_csv(input_file, header=None)

    split_columns_df = df[0].str.split('|', expand=True)

    # re-format the columns
    split_columns_df.columns = ['tick_id', 'price', 'volume', 'date', 'type']

    # re-order the columns
    split_columns_df = split_columns_df[['date', 'tick_id', 'price', 'volume', 'type']]

    split_columns_df.to_csv(output_file, index=False)

# split_csv("../../data/Tick/BTCGBP.csv", "preprocessed_BTCGBP.csv")

def display_dataset(dataset):
    # load dataset
    dataset = pd.read_csv(dataset, header=0, index_col=0)
    values = dataset.values
    groups = [1, 2]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
      pyplot.subplot(len(groups), 1, i)
      pyplot.plot(values[:, group])
      pyplot.title(dataset.columns[group], y=0.5, loc='right')
      i += 1
    pyplot.show()

display_dataset("preprocessed_BTCGBP.csv")



# # load dataset
# dataset = pd.read_csv("preprocessed_formatted_BTCGBP_1000_4.csv", header=0, index_col=0)
# dataset.drop('tick_id', axis=1, inplace=True) # seems irrelevent
# values = dataset.values

# # ensure all data is float
# values = values.astype('float32')