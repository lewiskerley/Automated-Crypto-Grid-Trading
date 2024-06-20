import pandas as pd
from numpy import array
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError


# pre-process into seperate columns:
def split_csv(input_file, output_file):
    df = pd.read_csv(input_file, header=None)

    split_columns_df = df[0].str.split('|', expand=True)

    # re-format
    split_columns_df.columns = ['tick_id', 'price', 'volume', 'date', 'type']

    df = pd.concat([split_columns_df, df[1]], axis=1)
    df.columns = ['tick_id', 'price', 'volume', 'date', 'type', 'grid_success']

    # re-order the columns
    df = df[['date', 'tick_id', 'price', 'volume', 'type', 'grid_success']]

    # wwrite the DataFrame to a new CSV file
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
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = pd.DataFrame(scaler.fit_transform(values))
scaled.columns = ["price", "volume", "type", "grid_success"]

print(scaled.head())



# split into train and test sets
values = scaled.values
k = 10
# number_of_ticks = 10
k_train_split = len(values) - round(len(values) / k)
train = values[:k_train_split, :]
test = values[k_train_split:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
train_y = array([[0, 1] if label == 1 else [1, 0] for label in train_y]) # one-hot encoding
test_X, test_y = test[:, :-1], test[:, -1]
test_y = array([[0, 1] if label == 1 else [1, 0] for label in test_y]) # one-hot encoding

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(2, activation='softmax')) # binary output
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())



def get_model(params, input_shape):
	model = Sequential()
	model.add(LSTM(units=params["lstm_units"], return_sequences=True, input_shape=input_shape))
	model.add(Dropout(rate=params["dropout"]))

	model.add(LSTM(units=params["lstm_units"], return_sequences=True))
	model.add(Dropout(rate=params["dropout"]))

	model.add(LSTM(units=params["lstm_units"], return_sequences=True))
	model.add(Dropout(rate=params["dropout"]))

	model.add(LSTM(units=params["lstm_units"], return_sequences=False))
	model.add(Dropout(rate=params["dropout"]))

	model.add(Dense(2, activation="softmax"))

	model.compile(loss=params["loss"],
              	optimizer=params["optimizer"],
              	metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

	return model

params = {
	"loss": "categorical_crossentropy",
	"optimizer": "adam",
	"dropout": 0.1,
	"lstm_units": 90,
	"epochs": 100,
	"batch_size": 8,
	"es_patience" : 10
}

model = get_model(params=params, input_shape=(train_X.shape[1], train_X.shape[2]))
print(model.summary())


# fit model
es_callback = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', patience=params["es_patience"])
history = model.fit(train_X, train_y, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[es_callback])

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()




# make a prediction
yhat = model.predict(test_X)
print(yhat[0:100])
print(test_y[0:100])

# evaluate
scores = model.evaluate(test_X, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# # calculate RMSE
# rmse = sqrt(mean_squared_error(yhat, test_X))
# print('Test RMSE: %.3f' % rmse)