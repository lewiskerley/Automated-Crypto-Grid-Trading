import pandas as pd
from numpy import array
from numpy import concatenate
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.optimizers.schedules import ExponentialDecay


def split_csv(input_file, output_file):
    df = pd.read_csv(input_file, header=None)

    split_columns_df = df[0].str.split('|', expand=True)

    # re-format
    split_columns_df.columns = ['tick_id', 'price', 'volume', 'date', 'type']

    df = pd.concat([split_columns_df, df[1]], axis=1)
    df.columns = ['tick_id', 'price', 'volume', 'date', 'type', 'grid_success']

    # re-order the columns
    df = df[['date', 'tick_id', 'price', 'volume', 'type', 'grid_success']]

    # write the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# split_csv("preprocessed_BTCGBP_1000_4.csv", "preprocessed_formatted_BTCGBP_1000_4.csv")

def display_dataset(dataset):
    # load dataset
    dataset = pd.read_csv(dataset, header=0, index_col=0)
    values = dataset.values
    groups = [1, 2, 4]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
      pyplot.subplot(len(groups), 1, i)
      pyplot.plot(values[:, group])
      pyplot.title(dataset.columns[group], y=0.5, loc='right')
      i += 1
    pyplot.show()

# display_dataset("preprocessed_formatted_BTCGBP_1000_4.csv")

# load dataset
dataset = pd.read_csv("preprocessed_formatted_BTCGBP_1000_4.csv", header=0, index_col=0)
dataset.drop('tick_id', axis=1, inplace=True) # seems irrelevent
values = dataset.values

# ensure all data is float
values = values.astype('float32')

# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = pd.DataFrame(scaler.fit_transform(values))
#scaled.columns = ["price", "volume", "type", "grid_success"]
#print(scaled.head())
values = pd.DataFrame(values)
values.columns = ["price", "volume", "type", "grid_success"]
print(values.head())




# new features:
# calculate the moving average of the 'price' feature
window_size = 10
window_size_slow = 100
values['price_ma'] = values['price'].rolling(window=window_size, min_periods=1).mean()
values['price_ma_slow'] = values['price'].rolling(window=window_size_slow, min_periods=1).mean()
values['momentum'] = values['price'] - values['price'].shift(10)
values.dropna(inplace=True)

# features and target data
# X = values[['price', 'volume', 'type']]
X = values[['volume', 'price_ma', 'price_ma_slow', 'momentum']]
y = values['grid_success']

scaler = MinMaxScaler(feature_range=(0, 1))
X = pd.DataFrame(scaler.fit_transform(X))

# reshape X to fit the input shape of LSTM
X = X.values.reshape(X.shape[0], 1, X.shape[1])

# training and test data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# the LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1, activation='sigmoid')
])

# compiling the model
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

# training
history = model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_val, y_val))

# evaluation
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {accuracy}')

# make predictions
predictions = model.predict(X_val)
print(y_val[-25:])
print(predictions[-25:])