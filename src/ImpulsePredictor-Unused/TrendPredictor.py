import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# using the twenty-ahead regression model
model = load_model('price_volume_wma_to_twenty-ahead_25epochs_0-001lr_1000batch_model.keras')


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

# evaluate the model
loss, mse, mae = model.evaluate(scaled_X, scaled_Y)
print(f'Validation MSE: {mse}, MAE: {mae}') # Validation MSE: 5.721857405660558e-07, MAE: 0.0004753757093567401