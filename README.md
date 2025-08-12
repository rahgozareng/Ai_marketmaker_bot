# AI Market-Maker Bot (PPO)

An automated market-making bot that uses a Reinforcement Learning agent (PPO) trained on orderbook data to decide quotes and order sizes.  
This project contains inference code to run a pre-trained model (`real_orderbook_rl_model_4`) and helper utilities for backtesting, simulation, and live execution.

---

## ⚙️ Overview
- Agent: **PPO** (Proximal Policy Optimization) — loaded via `stable_baselines3.PPO`.
- Purpose: Provide continuous bid/ask quotes on a selected trading pair and capture the spread while managing inventory and risk.
- Modes: **Backtest / Simulation** (historical orderbook) and **Live** (connect to exchanges via API).

---

## 🔥 Features
- RL-based decision making (load a pre-trained PPO model)
- Live execution using CEX APIs (ccxt compatible)
- Backtesting against historical orderbook snapshots
- Basic risk controls: position limits, cancel-on-exception, daily PnL caps
- Logging of orders, trades, and model actions
- Configurable environment parameters (grid size, observation window, reward shaping)

---

## 🧾 Requirements
- Python 3.9+ recommended
- Typical Python packages used by this project (put into `requirements.txt`):


---

## 🔧 Setup & Configuration
1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/ai-market-maker.git
cd ai-market-maker

2.Install dependecies

3. Add your trained model file to /models/
Example: models/real_orderbook_rl_model_4.zip (or .zip/.pt depending on how you saved it).

4. run



Running (Inference / Live)
Example (live mode):

bash
Copy code
python market_maker.py --mode live --model models/real_orderbook_rl_model_4
Example (backtest):

bash
Copy code
python market_maker.py --mode backtest --model models/real_orderbook_rl_model_4 --history data/orderbook_history.parquet
🧠 How inference works (high-level)
The environment (orderbook_env) converts recent orderbook snapshots into an observation vector.

The RL agent is loaded:

python
Copy code
from stable_baselines3 import PPO
model = PPO.load("models/real_orderbook_rl_model_4")
For each timestep the code calls:

python
Copy code
action, _ = model.predict(obs, deterministic=True)
action is decoded into market-making instructions (e.g., quote spread, order sizes, aggressive/ passive flag).

The execution module sends orders via your exchange_client (which wraps ccxt) and updates local state.

📈 Training (brief)
If you want to retrain or fine-tune:

python
Copy code
from stable_baselines3 import PPO
from envs.orderbook_env import OrderbookEnv
env = OrderbookEnv(...)  # wrap your historical data or simulator
model = PPO("MlpPolicy", env, verbose=1)

# Example training loop
model.learn(total_timesteps=1_000_000)
model.save("models/real_orderbook_rl_model_4")
Use vectorized environments, Monitor, and EvalCallback for checkpoints & evaluation.

Save frequently and keep the best checkpoint.

✅ Backtesting Guidelines
Always backtest on out-of-sample historical periods.

Model performance must be evaluated with realistic execution simulation: slippage, latency, order-fill probability, exchange fees.

Log per-trade PnL, drawdown, and inventory distribution.

⚠ Safety & Production Notes
Rate-limit API calls and use exponential backoff on errors.

Use safety checks: max_position, daily_loss_limit, circuit_breaker.

Run on a VPS with automatic restart and logging.

Consider an order-queue/ack system that tracks accepted/cancelled orders (don’t assume success).

📝 Troubleshooting
If the model throws shape errors: check the environment observation_space and ensure pre-processing (normalization) is identical to training.

If orders are not executed: verify API keys and that trading permission is enabled (no withdrawal permission needed).

Debugging tip: run in --mode dry to print actions without sending orders.

⚖ Disclaimer
This repository is for educational and experimental purposes only. Trading cryptocurrencies carries risk. The author is not responsible for financial loss or damages arising from the use of this software.

📬 Contact / Contribute
If you want help integrating another exchange, improving the env, or adding safety checks — open an issue or contact me at your.email@example.com.

License
MIT License — feel free to use, adapt, and improve.
