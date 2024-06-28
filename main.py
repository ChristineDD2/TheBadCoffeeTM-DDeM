from sqlite3 import connect
import time
from urllib import response
import ccxt
from logging_component import *
from private import PrivateParts
from execute_trade import TradersParadise
import random
import numpy as np
from public import PublicHorny
import pandas as pd
import requests
import pprint as pp
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create an instance of the ndax.io exchange
exchange = ccxt.cryptocom({
    # 'password': 'YOUR_API_PASSWORD',
    # Additional exchange-specific options if needed
}),

def connect_xmrig(miner, software):
    url = "./xmrig",
    connect_xmrig(miner=2, software="xmrig")
    connect == {
    response == requests.get(f'{url}', params = {
					
			"healthy-quantum": True,
			"coinor": 100,
			"guess-work": True,
			"school-hash": True
			}),
	}
while...:
    xmrig = {
        "authorized-seed-flare": True,
        "software": True,
        "response": True
    }
def calculate_sma(symbol, timeframe, period):
    # Get the OHLCV data for the specified symbol and timeframe
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=period)

    # Extract the closing prices from the OHLCV data
    close_prices = [candle[-1] for candle in ohlcv]

    # Calculate the simple moving average (SMA) for the closing prices
    sma = sum(close_prices) / len(close_prices)

    return sma


def calculate_rsi(symbol, timeframe, period):
    # Get the OHLCV data for the specified symbol and timeframe
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=period)

    # Extract the closing prices from the OHLCV data
    close_prices = np.array([candle[-1] for candle in ohlcv])

    # Calculate the RSI using the closing prices
    rsi = pd.RSI(close_prices, timeperiod=period)

    return rsi[-1]  # Get the latest RSI value


def execute_trade():
    # Get the list of available trading pairs
    markets = exchange.load_markets()

    for symbol in markets:
        try:
            # Get the latest ticker data
            ticker = exchange.fetch_ticker(symbol)
            logging.info(f"Ticker data for {symbol}: {ticker}")

            current_price = ticker['close']
            buy_price = current_price

            # Calculate sell prices
            sell_prices = [buy_price * (1 + percentage) for percentage in [0.0313, 0.0420, 0.060192]]  # Define multiple sell price levels

            # Calculate stop loss and take profit prices
            stop_loss_price = buy_price * 0.9750  # 1% lower than buy price
            take_profit_prices = [buy_price * (1 + percentage) for percentage in [0.0419, 0.0555, 0.1102]]  # Define multiple take profit levels

            # Check candlestick market data
            candle_data = exchange.fetch_ohlcv(symbol, '1m', limit=2)  # Fetch last 2 candles
            current_candle = candle_data[1]
            previous_candle = candle_data[-1]

            # Check buy condition based on candlestick market data
            if current_candle[1] > previous_candle[-1]:  # If current candle opens higher than the previous candle's close
                # Fetch account balance
                balance = exchange.fetch_balance()
                available_funds = balance['total']['USD']  # Assuming your base currency is USDT

               # Generate random amounts for buy and sell
                buy_amount = float(random.uniform(0.00002555, 100000000))  # Random amount between 1 and 100,000 as an integer

                # Place a market buy order with the random buy amount
                buy_order = exchange.create_market_buy_order(symbol, buy_amount)
                logger.info(f"Buy order placed for {symbol} at market price: {buy_order}")

                # Generate random sell amounts within a certain range
                sell_amounts = [np.random.uniform(0.00002555, 100000000, buy_amount) for _ in range(len(sell_prices))]

                # Place a market buy order with the dynamic buy amount
                buy_order = exchange.create_market_buy_order(symbol, buy_amount)
                logging.info(f"Buy order placed for {symbol} at market price: {buy_order}")

                # Place sell orders at specified prices with random sell amounts
                for sell_price, sell_amount in zip(sell_prices, sell_amounts):
                    sell_order = exchange.create_limit_sell_order(symbol, sell_amount, sell_price)
                    logging.info(f"Sell order placed for {symbol} at price: {sell_price}")

                # Set stop loss and take profit orders
                for take_profit_price in take_profit_prices:
                    exchange.create_order(
                        symbol,
                        'limit',
                        'sell',
                        buy_amount,
                        price=take_profit_price,
                        params={'takeProfit': True}
                    )
                    logging.info(f"Take profit order placed for {symbol} at price: {take_profit_price}")

                exchange.create_order(
                    symbol,
                    'stop',
                    'sell',
                    buy_amount,
                    stopPrice=stop_loss_price,
                    params={'stopLoss': True}
                )
                logging.info(f"Stop loss order placed for {symbol} at price: {stop_loss_price}")

        except ccxt.InsufficientFunds as e:
            logging.info(f"Insufficient funds for {symbol}. Skipping to the next trading pair.")

        except ccxt.BaseError as e:
            if "symbol" in str(e).lower():
                logging.info(f"Invalid symbol ({symbol}). Skipping to the next trading pair.")
            else:
                logging.info(f"An error occurred for {symbol}. Skipping to the next trading pair.")

        except Exception as e:
            logging.info(f"An error occurred: {str(e)}")

        time.sleep(5)

# Run the execute_trade() function indefinitely
while True:
    try:
        execute_trade()
    except Exception as e:
        logging.info(f"An error occurred: {str(e)}")

    time.sleep(1)  # 5 minutes interval
