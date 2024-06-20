import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

# load the dataset
data = pd.read_csv('grid_results_BTCGBP.csv', header=0)
data.dropna(subset=['grid_result'], inplace=True)

X = data.iloc[:, 1:-1].values
y = data['grid_result'].astype(int).values  # (grid_result)

# training and test set split at 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)


def train_lstm():
    # LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=16, input_shape=(X_train.shape[1], X_train.shape[2]), activation="relu", return_sequences=False))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # previous architecture
    # model.add(LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), activation="relu", return_sequences=True))
    # model.add(LSTM(units=128, activation="relu", return_sequences=True))
    # model.add(LSTM(units=64, activation="relu", return_sequences=False))
    # model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=1, epochs=500, class_weight={0: class_weights[0], 1: class_weights[1]})

    _, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy on test set: {accuracy}')


    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)



    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))


    for thresh in np.arange(0.1, 1.0, 0.1): # threshold searching, optimal: 0.5
        y_pred_binary = (y_pred > thresh).astype(int)
        print(f"Confusion Matrix {thresh}:")
        print(confusion_matrix(y_test, y_pred_binary))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))
    

    model.save('grid_trading_lstm_model.keras')

# train_lstm()

def lstm_grid_trader():
    tick_data = pd.read_csv('BTCGBP.csv')
    tick_data['date'] = pd.to_datetime(tick_data['date'], unit='ms')
    data_subset = tick_data

    agg_data = pd.read_csv('grid_results_BTCGBP.csv')
    agg_data['date'] = pd.to_datetime(agg_data['date'])

    model = load_model('grid_trading_lstm_model.keras')
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))

    Y_prediction = model.predict(X_s)

    level_outcome = []

    for i in range(0, len(agg_data) - 5):
        start_time = agg_data.at[i, 'date']
        if i < len(agg_data) - 30:
            end_time = agg_data.at[i + 30, 'date'] # a month limit
        else:
            end_time = tick_data['date'].max()  # go to the end of tick data

        #print(Y_prediction[i])
        if (Y_prediction[i] > 0.5):
            # find the tick row just after the start_time
            just_after_start_time = tick_data[tick_data['date'] > start_time].iloc[0]['date']
            data_subset = data_subset[(data_subset['date'] > just_after_start_time)]
            data_subset_to_use = data_subset[(data_subset['date'] <= end_time)]
            
            result = simulate_algorithm(data_subset_to_use)
            if result == 4:
                level_outcome.append("Unknown") # outcome extended past timeout
            else:
                level_outcome.append(result)
        else:
            # grid trading scenario prediction deemed unsuccessful - continue 
            level_outcome.append("Pass")
        
    results_csv = pd.DataFrame(level_outcome, columns=["LevelOutcome"])
    results_csv.to_csv("lstm_results.csv", index=False)



def simulate_algorithm(period_data):
    starting_price = float(period_data.iloc[0]['price'])
    print(period_data.iloc[0]['date'])

    price_interval = 600
    next_upper = starting_price + price_interval
    next_lower = starting_price - price_interval
    moves = []

    for index, tick in period_data.iterrows():
        price = float(tick['price'])

        if price >= next_upper:
            next_upper += price_interval
            next_lower += price_interval
            moves.append(True)

            if moves == [True, True, True, True]: # the four symmetric scenarios posed by a Level-3 grid system
              return 3
            elif moves == [False, True]:
                return 0
            elif moves == [False, False, True]:
                return -1
            elif moves == [False, False, False, True]:
                return -2
        elif price <= next_lower:
            next_upper -= price_interval
            next_lower -= price_interval
            moves.append(False)

            if moves == [True, False]: # the four symmetric scenarios posed by a Level-3 grid system
                return 0
            elif moves == [True, True, False]:
                return 1
            elif moves == [True, True, True, False]:
                return 2
            elif moves == [False, False, False, False]:
                return -3
    return 4
            
lstm_grid_trader()