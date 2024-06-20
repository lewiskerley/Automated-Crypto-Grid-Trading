import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam


df = pd.read_csv("preprocessed_BTCGBP_wma_twenty.csv", header=0)
X = df[["price", "volume", "wma"]].values
Y = df['price_twenty_ahead'].values

# scale input features
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler_X.fit_transform(X)
scaled_X = scaled_X.reshape(scaled_X.shape[0], 1, scaled_X.shape[1])

# scale target
scaler_Y = MinMaxScaler(feature_range=(0, 1))
scaled_Y = scaler_Y.fit_transform(Y.reshape(-1, 1))
scaled_Y = scaled_Y.reshape(-1, 1)

# print("Predicted", scaler_Y.inverse_transform(np.array([0.6528591]).reshape(-1, 1)))
# print("Actual", scaler_Y.inverse_transform(np.array([0.65859916]).reshape(-1, 1)))



# split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(scaled_X, scaled_Y, test_size=0.2, random_state=42, shuffle=False)
# print(X_train[:30])
# print(y_train[:30])

# the LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation="relu"),
    Dense(1, activation='relu')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

# training
history = model.fit(X_train, y_train, epochs=25, batch_size=1000, validation_data=(X_val, y_val))

# evaluation
loss, mse, mae = model.evaluate(X_val, y_val)
print(f'Validation MSE: {mse}, MAE: {mae}')

# predictions
predictions = model.predict(X_val)
print(y_val[-25:])
print(predictions[-25:])

print("Actual", scaler_Y.inverse_transform(y_val[-25:]))
print("Predicted", scaler_Y.inverse_transform(predictions[-25:]))

model.save("price_volume_wma_to_twenty-ahead_25epochs_0-001lr_1000batch_model.keras") # optimal parameters