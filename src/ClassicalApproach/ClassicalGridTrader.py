import pandas as pd
import math

# Load the CSV data into a DataFrame
data = pd.read_csv("../../data/Tick/BTCGBP.csv", header=None)
print("Example data:\n", data.head(2))

def simple_grid_trading(tick_data, price_interval = 500, rounded=False):
    starting_price = float((tick_data.iloc[0].values[0]).split("|")[1])
    if rounded: # nearest thousand
        starting_price = round(starting_price, -3)
        if price_interval % 1000 != 0:
            print("Error, price_interval must be a multiple of 1000")
            return
    next_upper = starting_price + price_interval
    next_lower = starting_price - price_interval
    moves = []
    print("Initial Price:", starting_price)


    for index, tick in tick_data.iterrows():
        tick_data_parts = tick.values[0].split("|")
        price = float(tick_data_parts[1])

        if price >= next_upper:
            next_upper += price_interval
            next_lower += price_interval
            moves.append(True)
        elif price <= next_lower:
            next_upper -= price_interval
            next_lower -= price_interval
            moves.append(False)

        if index % 5000000 == 0:
            print(f"{index / len(tick_data) * 100}%")

    
    total_profit = 0
    total_profit_types = [0,0,0,0]
    current_sequence = []
    level_outcome = []

    for move in moves[1:]: # Filter out the first element
        current_sequence.append(move)

        if current_sequence == [True, False]: # the four symmetric scenarios posed by a Level-3 grid system
            total_profit += 100
            total_profit_types[0] += 1
            level_outcome.append(0)
            current_sequence = []
        elif current_sequence == [True, True, False]:
            total_profit += 100
            total_profit_types[1] += 1
            level_outcome.append(1)
            current_sequence = []
        elif current_sequence == [True, True, True, False]:
            total_profit += 0
            total_profit_types[2] += 1
            level_outcome.append(2)
            current_sequence = []
        elif current_sequence == [True, True, True, True]:
            total_profit -= 600
            total_profit_types[3] += 1
            level_outcome.append(3)
            current_sequence = []

        elif current_sequence == [False, True]: # the four symmetric scenarios posed by a Level-3 grid system
            total_profit += 100
            total_profit_types[0] += 1
            level_outcome.append(0)
            current_sequence = []
        elif current_sequence == [False, False, True]:
            total_profit += 100
            total_profit_types[1] += 1
            level_outcome.append(-1)
            current_sequence = []
        elif current_sequence == [False, False, False, True]:
            total_profit += 0
            total_profit_types[2] += 1
            level_outcome.append(-2)
            current_sequence = []
        elif current_sequence == [False, False, False, False]:
            total_profit -= 600
            total_profit_types[3] += 1
            level_outcome.append(-3)
            current_sequence = []

        if len(current_sequence) > 10:
            print("ERROR: ", current_sequence)

    print(f"Total profit: {total_profit}, for interval: {price_interval}")
    print(f"Profits type(100, 100, 0 -600): {total_profit_types}")

    cidf = pd.DataFrame(level_outcome, columns=["LevelOutcome"])
    if rounded:
        cidf_filename = f"classical_interval_{price_interval}_rounded.csv"
    else:
        cidf_filename = f"classical_interval_{price_interval}.csv"
    cidf.to_csv(cidf_filename, index=False)



#simple_grid_trading(data, 1000, rounded=True)

# for i in range(100, 2500, 100):
#     print(i)
#     simple_grid_trading(data, i)



simple_grid_trading(data, 500) # optimal grid spacing: 600