import csv
from socket import timeout
from symtable import Symbol
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
import datetime
import sys
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from collections import deque


MEXC_API_KEY = "xxx"
MEXC_API_SECRET = "xxx"

SYMBOL = "BTCUSDC"
SPREAD = 0.08  # Adjust spread percentage (0.05 = 0.05% spread)

ORDER_SIZE = 0.00002  # Amount of BTC per order
SLEEP_TIME = 10  # Seconds between updates

price_history = deque(maxlen=2000)

# Settings
CHECK_INTERVAL = 10  # Seconds between quick checks

#global catch error
logging.basicConfig(filename="bot_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Logging setup
logging.basicConfig(filename="bot_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def log_exception(exctype, value, tb):
    logging.error("Uncaught Exception:", exc_info=(exctype, value, tb))
sys.excepthook = log_exception


# API Functions
def get_mexc_server_time():
    try:
        response = requests.get("https://api.mexc.com/api/v3/time", timeout=5)
        return response.json().get("serverTime", int(time.time() * 1000))
    except requests.RequestException as e:
        logging.error(f"MEXC server time fetch failed: {e}")
        return int(time.time() * 1000)

def get_mexc_price():
    url = "https://api.mexc.com/api/v3/depth?symbol=BTCUSDC&limit=100"
    
    try:
        response = requests.get(url, timeout=5)  # Timeout set to 5 seconds
        response.raise_for_status()  # Raises an error for HTTP codes 4xx/5xx
        data = response.json()
        if not data.get("bids") or not data.get("asks"):
            logging.error("Empty order book received.")
            return None, None
        best_bid = float(data['bids'][0][0])  # Highest buy price
        best_ask = float(data['asks'][0][0])  # Lowest sell price
        all_Orderbooks = data
        return best_bid, best_ask , all_Orderbooks
    except Exception as e:
        logging.error(f"Failed to fetch Mexc price: {e}")
        return None, None

# Function to place an order on MEXC
def place_mexc_order(side, price,):
    url = "https://api.mexc.com/api/v3/order"
    
    params = {
        "symbol": SYMBOL,
        "side": side,
        "type": "LIMIT",
        "quantity": ORDER_SIZE,
        "price": price,
        "timeInForce": "GTC",
        "timestamp": get_mexc_server_time(),
        "recvWindow": 5000  # Increase recvWindow (default is 500ms)
    }

    query_string = urlencode(params)
    signature = hmac.new(MEXC_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    
    params["signature"] = signature  # Add signature to params
    
    headers = {
        "X-MEXC-APIKEY": MEXC_API_KEY  # Ensure correct API Key header
    }
    try:
      response = requests.post(url, headers=headers, params=params)  # Send as params, not JSON
      response.raise_for_status() 
      #print("MEXC response:", response.status_code, response.text)
      return response.json()
    except Exception as e:
      logging.error(f"MEXC order failed: {e}")
      return None
  
  
def sign_request(params):
    """Sign the API request using HMAC-SHA256."""
    query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
    signature = hmac.new(MEXC_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return f"{query_string}&signature={signature}"
  
        
def cancel_all_orders():
    url = f"https://api.mexc.com/api/v3/openOrders?symbol={SYMBOL}"
    params = {"symbol": SYMBOL, "timestamp":get_mexc_server_time()}
    url = f"https://api.mexc.com/api/v3/openOrders?{sign_request(params)}"
    try:
        response = requests.delete(url, headers={"X-MEXC-APIKEY": MEXC_API_KEY})
        response.raise_for_status()  # Raise an error for 4xx/5xx status codes
        logging.debug("All orders canceled successfully.")
        return "success"
    except Exception as e:
        logging.error(f"Failed to cancel orders: {e}")
        return "failed"


def get_last_mexc_trades():
    """Fetch only new executed trades from MEXC."""
    url = "https://api.mexc.com/api/v3/myTrades"
    
    params = {
        "symbol": SYMBOL,
        "timestamp": get_mexc_server_time(),
        "limit": 10,  # Get the latest 50 trades (adjustable)
    }

    query_string = urlencode(params)
    signature = hmac.new(MEXC_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    headers = {"X-MEXC-APIKEY": MEXC_API_KEY}
    try:
        response = requests.get(url, headers=headers, params=params,timeout=5)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        logging.error(f"Failed to fetch MEXC trade history: {e}")
        print(f"Failed to fetch MEXC trade history: {e}")
        return []


def GetOpenOrders():
    """Fetch only new executed trades from MEXC."""
    url = "https://api.mexc.com/api/v3/openOrders"
    
    params = {
        "symbol": SYMBOL,
        "timestamp": get_mexc_server_time()        
    }

    query_string = urlencode(params)
    signature = hmac.new(MEXC_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    headers = {"X-MEXC-APIKEY": MEXC_API_KEY}
    try:
        response = requests.get(url, headers=headers, params=params,timeout=5)
        response.raise_for_status()
        return response.json()

    except Exception as e:
        logging.error(f"Failed to fetch open orders: {e}")
        print(f"Failed to fetch open orders: {e}")
        return None

def normalize(value, min_value, max_value, epsilon=1e-8):
        """Normalize a value to the range [-1, 1]."""
        return 2 * ((value - min_value) / (max_value - min_value + epsilon)) - 1


model = PPO.load("real_orderbook_rl_model_4")  # Use the best model you found

# Run the arbitrage checker in a loop
while True:
    try:      
      mexc_bid, mexc_ask, all_Orderbooks = get_mexc_price()
      if mexc_bid is None or mexc_ask is None:
        print("Skipping check due to MEXC API failure.")
        time.sleep(SLEEP_TIME)
        continue
      
      bid_volume = sum(float(bid[1]) for bid in all_Orderbooks["bids"])
      ask_volume = sum(float(ask[1]) for ask in all_Orderbooks["asks"])  
      relative_spread = (mexc_ask - mexc_bid) / ((mexc_ask + mexc_bid)/2)
      price_history.append((mexc_ask, mexc_bid, ask_volume, bid_volume , relative_spread))
      
      if len(price_history) < 100:  # Wait for 100 samples
        print("waiting for 100 samples")
        time.sleep(SLEEP_TIME)
        continue
      
      max_spread = round(mexc_ask * 0.0025, 0)  # Maximum possible spread adjustment
      min_spread = round(mexc_ask * 0.0001, 0)
      SPREAD = mexc_ask - mexc_bid

      # Extract features using zip
      asks, bids,ask_vols, bid_vols,  relatives = zip(*price_history)      
      min_ask, max_ask = min(asks), max(asks)
      min_bid, max_bid = min(bids), max(bids)
      min_ask_volume, max_ask_volume = min(ask_vols), max(ask_vols)
      min_bid_volume, max_bid_volume = min(bid_vols), max(bid_vols)
      min_relative_spread, max_relative_spread = min(relatives), max(relatives)
           
      # Normalize values
      normalized_best_bid = normalize(mexc_bid, min_bid, max_bid)
      normalized_best_ask = normalize(mexc_ask, min_ask, max_ask)
      normalized_spread = normalize(SPREAD, min_spread, max_spread)
      normalized_relative_spread = normalize(relative_spread, min_relative_spread, max_relative_spread)  # Relative spread is small
      normalized_ask_volume = normalize(ask_volume, min_ask_volume, max_ask_volume)
      normalized_bid_volume = normalize(bid_volume, min_bid_volume, max_bid_volume)
  
      orders_filled = GetOpenOrders()
      if orders_filled is None:
          print("Skipping check due to MEXC API open orders failure.")
          time.sleep(SLEEP_TIME)
          continue
      # Prepare observation (normalize if required)
      observation = np.array([
          normalized_best_bid,
          normalized_best_ask,
          normalized_ask_volume,
          normalized_bid_volume,
          normalized_spread,
          normalized_relative_spread,
          len(orders_filled) / 2,  # Number of open orders
      ])
      # Get action from model
      action, _states = model.predict(observation, deterministic=True)
      

      # Scale spread adjustment (-max_spread, +max_spread)
      new_spread = action[0]  * max_spread  
      if new_spread < min_spread: new_spread = min_spread
      mid_price = (mexc_bid + mexc_ask) / 2
      buy_price = mid_price - new_spread
      sell_price = mid_price + new_spread
    
      orders_filled = GetOpenOrders()
      # Map order action to discrete values
      order_action_continuous = action[1]  
      if order_action_continuous < 0.50:
          action_type = 0  # "keep"
      elif 0.50 <= order_action_continuous:
          action_type = 1  # "modify"
        
      if action_type == 1 or len(orders_filled) == 0:
        cancel_response = cancel_all_orders()  # Remove old orders
        if(cancel_response == "failed"):
          print("failed to calcel orders")
          time.sleep(SLEEP_TIME)
          continue
    
        place_mexc_order("SELL", sell_price)
        place_mexc_order("BUY", buy_price)      
        print(f"updated orders: Buy at {buy_price}, Sell at {sell_price}")
        print(f"Action: {action}")
       
    except Exception as e:
        logging.error(f"Error in main loop check: {e}")
   
    time.sleep(SLEEP_TIME)

 




