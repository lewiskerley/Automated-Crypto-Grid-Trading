# Automated-Crypto-Grid-Trading

Grid trading is a systematic methodology which may be employed within a
crypto-currency market, utilising predefined price-level intervals to buy and sell
orders in an established grid-like pattern strategically. A high-level understand-
ing of grid trading can be obtained by drawing horizontal lines every X amount
of units above and below an asset’s price level. Taking X to be 100 units and
an initial price level of 1.2400 the grid system begins to place one buy and one
sell order at: 1.2400, 1.2500, 1.2600 and so on. And further orders below the
price level at: 1.2400, 1.2300, 1.2200 and so on. Every time a new price level is
achieved, new buy and sell order is entered at that level.

The results of this software indicate that the Artificial Neural Network (ANN)
approach significantly outperforms the classical grid trading approach in the
context of BTCGBP cryptocurrency trading. The ANN method achieved a re-
turn on investment (ROI) of 69.1667%, greater than the classical baseline of
24.6667% and much greater than the initial S&P 500 baseline at 10.47%.
Additionally, the volatility and maximum drawdown metrics were reduced low-
ering the risk exposure to just 6% of the account. The Sharpe ratio increased
from 0.1135 to 0.4452 showing a much more balanced risk-return metric.
Despite this improvement, the ANN model did not beat the S&P 500’s Sharpe
ratio of approximately 2.3 which suggests that this strategy is much higher risk
at the gain of a much greater ROI. Both the classical and ANN approaches
consistently generate approximately one trade per day, demonstrating a robust
and reliable strategy compared to those that trade infrequently.


# Usage

1. Install the required Python Libraries:
```
pip install -r requirements.txt
```

2. Run any of the main Scripts:
```
python src/ClassicalApproach/ClassicalGridTrader.py
python src/ANN/ANNGridTrading.py
python src/LSTM/LSTMGridTrading.py
```

3. Monitor the results in the console or output files