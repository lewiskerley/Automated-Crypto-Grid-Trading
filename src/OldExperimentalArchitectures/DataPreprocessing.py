import pandas as pd
import numpy as np
import math
import time

# Load the CSV data into a DataFrame
data = pd.read_csv("../../data/Tick/BTCGBP.csv", header=None)
print("Example data:\n", data.head(2))

price_interval = 1000 # £1000
max_drawdown_level = 4 # Level 4

tick_count_interval = 10000 # Only mark every 10000 ticks



# Assuming a £1000 (by inspection) fixed grid initiating at the price of the current tick data
# to output a result indicating the success of a grid trade setup.
# Level 4 = Failure, Breakeven / profit after initial entry = Success.

outcome_per_tick = np.empty((int(len(data) / tick_count_interval) + 1, 1))
start_time = time.time()
print("Outcome shape:", outcome_per_tick.shape)

# for i, tick in data.iterrows():
for i in range(0, len(data), tick_count_interval):
    new_index = int(i / tick_count_interval)
    tick = data.iloc[i]
    tick_data_parts = tick.values[0].split("|")
    tick_id = int(tick_data_parts[0])
    price = float(tick_data_parts[1])
    #quantity = float(tick_data_parts[2])
    #timestamp = int(tick_data_parts[3])
    #is_sell = bool(tick_data_parts[4])

    last_entry = price
    next_high = price + price_interval
    next_low = price - price_interval
    grid_cashed_out_profit = 0
    grid_in_play_buys = []
    grid_in_play_sells = []
    grid_returns = 0
    grid_finished = False


    # Run a grid trading algorithm starting from tick (i)
    # print("Running grid trade algorithm from tick", tick_id)
    for j, tick in data.iloc[i+1:].iterrows():
        grid_tick_data_parts = data.iloc[j, 0].split("|")
        grid_tick_id = int(grid_tick_data_parts[0])
        grid_tick_price = float(grid_tick_data_parts[1])
        # print("Price:", grid_tick_price)

        while grid_tick_price >= next_high:
            # print("Up")
            grid_cashed_out_profit += price_interval
            grid_in_play_sells.append(0)
            for s_id in range(len(grid_in_play_sells)):
                grid_in_play_sells[s_id] -= price_interval

            for b_id in range(len(grid_in_play_buys)):
                grid_in_play_buys[b_id] += price_interval

            next_high += price_interval
            last_entry += price_interval

            # print(f"Tick {tick_id}, cashed out: {grid_cashed_out_profit}, in play: {sum(grid_in_play_buys) + sum(grid_in_play_sells)}, price: {grid_tick_price}")

            total = grid_cashed_out_profit + (sum(grid_in_play_buys) + sum(grid_in_play_sells))
            # print(total, len(grid_in_play_sells), len(grid_in_play_buys))
            if total >= 0 and (len(grid_in_play_sells) + len(grid_in_play_buys) > 1):
                grid_finished = True # Exit the grid trading algorithm loop with a success
                grid_returns = total
                break
            elif (len(grid_in_play_sells) >= max_drawdown_level or len(grid_in_play_buys) >= max_drawdown_level):
                grid_finished = True # Exit the grid trading algorithm loop with a failure
                grid_returns = total
                break

        if grid_finished:
            break

            
        while grid_tick_price <= next_low:
            # print("Down")
            grid_cashed_out_profit += price_interval
            grid_in_play_buys.append(0)
            for b_id in range(len(grid_in_play_buys)):
                grid_in_play_buys[b_id] -= price_interval

            for s_id in range(len(grid_in_play_sells)):
                grid_in_play_sells[s_id] += price_interval

            next_low -= price_interval
            last_entry -= price_interval

            # print(f"Tick {tick_id}, cashed out: {grid_cashed_out_profit}, in play: {sum(grid_in_play_buys) + sum(grid_in_play_sells)}, price: {grid_tick_price}")

            total = grid_cashed_out_profit + (sum(grid_in_play_buys) + sum(grid_in_play_sells))
            if total >= 0 and (len(grid_in_play_sells) + len(grid_in_play_buys) > 1):
                grid_finished = True # Exit the grid trading algorithm loop with a success
                grid_returns = total
                break
            elif (len(grid_in_play_sells) >= max_drawdown_level or len(grid_in_play_buys) >= max_drawdown_level):
                grid_finished = True # Exit the grid trading algorithm loop with a failure
                grid_returns = total
                break

        if grid_finished:
            break
    

    # Outcome: 1 is positive result (or break even), 0 otherwise
    outcome_per_tick[new_index] = int(grid_returns >= 0)
    # print(f"Outcome for tick {i}:", outcome_per_tick[new_index])

    if ((new_index+1) % 4 == 0):
        print(f"Tick: {tick_id} ({new_index + 1} / {int(len(data) / tick_count_interval)}), Outcome for algorithm: {outcome_per_tick[new_index]}, Elapsed: {round((time.time() - start_time) * 100) / 100}s")
        #i = len(data)
        #break

    if ((new_index+1) % 40 == 0):
        # Save the preprocessed data locally MIDWAY
        corresponding_data = data.iloc[::tick_count_interval]
        preprocessed_data = pd.concat([corresponding_data.reset_index(drop=True), pd.DataFrame(outcome_per_tick)], axis=1)
        preprocessed_data.to_csv("checkpoint_preprocessed_BTCGBP_1000_4.csv", index=False, header=False)

# Save the preprocessed data locally
corresponding_data = data.iloc[::tick_count_interval]
preprocessed_data = pd.concat([corresponding_data.reset_index(drop=True), pd.DataFrame(outcome_per_tick)], axis=1)
preprocessed_data.to_csv("preprocessed_BTCGBP_1000_4.csv", index=False, header=False)


# (Check) Read the preprocessed data and labels
preprocessed_data = pd.read_csv("preprocessed_BTCGBP_1000_4.csv", header=None)
train_data = preprocessed_data.iloc[:, 0].values
train_labels = preprocessed_data.iloc[:, 1].values

print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)

print("Example pre-processed data:\n", preprocessed_data.head(20))
# print("Example Train data:", train_data[0:20])
# print("Example Label data:", train_labels[0:20])