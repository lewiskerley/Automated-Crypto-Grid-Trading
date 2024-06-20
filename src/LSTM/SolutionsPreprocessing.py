from math import floor
import pandas as pd

# pre-processing the grid_success column for further usage in training the LSTM

tick_data = pd.read_csv('BTCGBP.csv')
tick_data['date'] = pd.to_datetime(tick_data['date'], unit='ms')

#agg_data = pd.read_csv('preprocessed_BTCGBP.csv')
agg_data = pd.read_csv('grid_results_BTCGBP.csv')
agg_data['date'] = pd.to_datetime(agg_data['date'])

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
              return False
            elif moves == [False, True]:
                return True
            elif moves == [False, False, True]:
                return True
            elif moves == [False, False, False, True]:
                return True
        elif price <= next_lower:
            next_upper -= price_interval
            next_lower -= price_interval
            moves.append(False)

            if moves == [True, False]: # the four symmetric scenarios posed by a Level-3 grid system
                return True
            elif moves == [True, True, False]:
                return True
            elif moves == [True, True, True, False]:
                return True
            elif moves == [False, False, False, False]:
                return False

    # current_sequence = []

    # for move in moves:
    #     current_sequence.append(move)

    #     if current_sequence == [True, False]:
    #         return True
    #     elif current_sequence == [True, True, False]:
    #         return True
    #     elif current_sequence == [True, True, True, False]:
    #         return True
    #     elif current_sequence == [True, True, True, True]:
    #         return False

    #     elif current_sequence == [False, True]:
    #         return True
    #     elif current_sequence == [False, False, True]:
    #         return True
    #     elif current_sequence == [False, False, False, True]:
    #         return True
    #     elif current_sequence == [False, False, False, False]:
    #         return False

    #     if len(current_sequence) > 10:
    #         print("ERROR: ", current_sequence)
    #         return False
        
    print("Incomplete", moves)
    return False
    

#agg_data['grid_result'] = None
data_subset = tick_data

# print(f"{(batch - 1) * floor(len(agg_data)/10)}, {batch * floor(len(agg_data)/10)}")
for i in range(0, len(agg_data)):
    start_time = agg_data.at[i, 'date']
    if i < len(agg_data) - 30:
        end_time = agg_data.at[i + 30, 'date'] # a month limit
    else:
        end_time = tick_data['date'].max()  # go to the end of tick data

    # find the tick row just after the start_time
    just_after_start_time = tick_data[tick_data['date'] > start_time].iloc[0]['date']
    data_subset = data_subset[(data_subset['date'] > just_after_start_time)]
    data_subset_to_use = data_subset[(data_subset['date'] <= end_time)]
    
    result = simulate_algorithm(data_subset_to_use)
    
    agg_data.at[i, 'grid_result'] = result

agg_data.to_csv('grid_results_BTCGBP.csv', index=False)
