import pandas as pd
import math

# load the CSV data into a DataFrame
data = pd.read_csv("../../data/Tick/BTCGBP.csv", header=None)
print("Example data:\n", data.head(2))


# experiment: No-stop no-level-limit model
def simple_grid_trading(tick_data, price_interval = 100):
    starting_price = float((tick_data.iloc[0].values[0]).split("|")[1])
    print("Initial Price:", starting_price)


    last_interval_price = starting_price

    streak_cashed_out_profit = 0
    streak_in_play_buys = []
    streak_in_play_sells = []

    total_profit = 0
    max_drawdown = 0

    for _, tick in tick_data.iterrows():
        tick_data_parts = tick.values[0].split("|")
        tick_id = int(tick_data_parts[0])
        price = float(tick_data_parts[1])
        #quantity = float(tick_data_parts[2])
        #timestamp = int(tick_data_parts[3])
        #is_sell = bool(tick_data_parts[4])

        above_price_level = last_interval_price + price_interval
        below_price_level = last_interval_price - price_interval
        while price >= above_price_level:
            # print("Up")
            streak_cashed_out_profit += price_interval
            streak_in_play_sells.append(0)
            for i in range(len(streak_in_play_sells)):
                streak_in_play_sells[i] -= price_interval

            for i in range(len(streak_in_play_buys)):
                streak_in_play_buys[i] += price_interval

            above_price_level += price_interval
            last_interval_price += price_interval

            # print(f"Tick {tick_id}, cashed out: {streak_cashed_out_profit}, in play: {sum(streak_in_play_buys) + sum(streak_in_play_sells)}, price: {price}")

            total = streak_cashed_out_profit + (sum(streak_in_play_buys) + sum(streak_in_play_sells))
            if total >= 0 and (len(streak_in_play_sells) + len(streak_in_play_buys) > 1):
                print(f"Tick {tick_id}, Success - Cashed out all trades with a total of {total}!")
                streak_cashed_out_profit = 0
                streak_in_play_buys = []
                streak_in_play_sells = []
                total_profit += total
            elif total < max_drawdown:
                max_drawdown = total

            
        while price <= below_price_level:
            # print("Down")
            streak_cashed_out_profit += price_interval
            streak_in_play_buys.append(0)
            for i in range(len(streak_in_play_buys)):
                streak_in_play_buys[i] -= price_interval

            for i in range(len(streak_in_play_sells)):
                streak_in_play_sells[i] += price_interval

            below_price_level -= price_interval
            last_interval_price -= price_interval

            # print(f"Tick {tick_id}, cashed out: {streak_cashed_out_profit}, in play: {sum(streak_in_play_buys) + sum(streak_in_play_sells)}, price: {price}")

            total = streak_cashed_out_profit + (sum(streak_in_play_buys) + sum(streak_in_play_sells))
            if total >= 0 and (len(streak_in_play_sells) + len(streak_in_play_buys) > 1):
                print(f"Tick {tick_id}, Success - Cashed out all trades with a total of {total}!")
                streak_cashed_out_profit = 0
                streak_in_play_buys = []
                streak_in_play_sells = []
                total_profit += total
            elif total < max_drawdown:
                max_drawdown = total
            
        # if tick_id > 500:
        #     break

    return total_profit, max_drawdown
        


        
        
outcome, max_drawdown = simple_grid_trading(data)
print(f"Final result - Outcome: {outcome}, Drawdown: {max_drawdown}")