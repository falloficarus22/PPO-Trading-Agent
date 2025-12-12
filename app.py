import torch
import numpy as np
import pandas as pd
import ccxt
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import *
from src.environment import TradingEnvironment
from src.agent import PPOAgent

app = FastAPI(title="PPO Live Trader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
LIVE_TRADING = True        # Set to False to go back to Backtest Simulation
REFRESH_RATE = 2.0         # Seconds to wait between Live checks (poll rate)
EXCHANGE_LIMIT = 100       # How many candles to fetch for context

# --- GLOBAL STATE ---
exchange = ccxt.binance({'enableRateLimit': True})
agent = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paper Trading State
paper_account = {
    "balance": INITIAL_BALANCE,
    "position": 0.0,
    "entry_price": 0.0,
    "portfolio_value": INITIAL_BALANCE
}

# In-Memory Data Buffer
live_df = None
last_processed_time = 0

@app.on_event("startup")
async def startup_event():
    global agent, device
    print(f"Initializing Live Trader on {device}...")
    
    # Load Agent
    # Note: We need to know dimensions. For standard Env(Window=50), Obs= (50, 12).
    # We assume these fixed dimensions based on your training.
    # Obs Shape: Window(50) * Features(12) = 600
    OBS_DIM = WINDOW_SIZE * 12 
    ACTION_DIM = 3
    
    agent = PPOAgent(OBS_DIM, ACTION_DIM, device=device)
    
    model_path = MODELS_DIR / 'ppo_final.pth'
    if model_path.exists():
        agent.load(model_path)
        agent.policy.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: No model found at {model_path}")

    print("System Ready for Live Data.")

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/step")
async def step():
    global live_df, last_processed_time, paper_account
    
    try:
        # 1. Fetch Live Data
        # We fetch slightly more than WINDOW_SIZE to allow for SMC lookahead NaNs
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=EXCHANGE_LIMIT)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 2. Check if we have a NEW candle or if we just want to update current price
        # For this demo, we will process every request as a potential "Step" decision,
        # but in production you'd only act on candle Close. 
        # We'll use the latest candle (even if open) to get "Real Time" feel.
        
        # 3. Process Features (SMC)
        # We create a temporary Env to reuse the feature engineering logic
        # This is inefficient but ensures exact consistency with training code.
        temp_env = TradingEnvironment(df, window_size=WINDOW_SIZE)
        
        # 4. Get Observation
        # The environment usually randomly picks a step in reset(). 
        # We need the *Latest* step.
        # We manually construct the observation for the last valid window.
        
        # Due to SMC "lookahead" (shift -3), the last 3 rows might have NaNs for Swing/OBs.
        # We take the window ending at -1 (Current) anyway, assuming NaNs = 0 (handled in Env).
        
        # Extract features from the internal DF of the env (which has SMC columns)
        # We need custom access to `_get_observation` logic without `reset()`.
        # Let's mock the env state to point to the end.
        temp_env.current_step = len(temp_env.df) - 1
        
        # Manually invoke logic similar to _get_observation but updating dynamic attributes
        temp_env.balance = paper_account["balance"]
        temp_env.position = paper_account["position"]
        
        obs = temp_env._get_observation()
        
        # 5. Agent Decision
        action = agent.get_action(obs, training=False)
        
        # 6. Execute "Paper Trade"
        current_price = df.iloc[-1]['close']
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_str = action_map.get(action, 'UNKNOWN')
        
        execute_paper_trade(action, current_price)
        
        # 7. Prepare Response
        row = df.iloc[-1]
        candle = {
            "time": int(row['timestamp'].timestamp()),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
        }

        return {
            "step": len(df), # Just a counter proxy
            "candle": candle,
            "action": action_str,
            "balance": float(paper_account["balance"]),
            "position": float(paper_account["position"]),
            "portfolio_value": float(paper_account["portfolio_value"]),
            "mode": "LIVE"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def execute_paper_trade(action, price):
    global paper_account
    
    bal = paper_account["balance"]
    pos = paper_account["position"]
    entry = paper_account["entry_price"]
    fee_rate = TRADING_FEE
    
    # 0=HOLD, 1=BUY, 2=SELL
    if action == 1: # BUY
        if pos == 0:
            # Open Long
            pos = bal / price
            bal -= bal * fee_rate
            entry = price
        elif pos < 0:
            # Close Short, Open Long
            # 1. Close Short
            pnl = abs(pos) * (entry - price)
            fee = abs(pos) * price * fee_rate
            bal += pnl - fee
            pos = 0
            
            # 2. Open Long
            pos = bal / price
            fee = bal * fee_rate
            bal -= fee
            entry = price
            
    elif action == 2: # SELL
        if pos == 0:
            # Open Short
            pos = -(bal / price)
            bal -= bal * fee_rate
            entry = price
        elif pos > 0:
            # Close Long, Open Short
            # 1. Close Long
            pnl = pos * (price - entry)
            fee = pos * price * fee_rate
            bal += pnl - fee
            pos = 0
            
            # 2. Open Short
            pos = -(bal / price)
            fee = abs(pos) * price * fee_rate
            bal -= fee
            entry = price
            
    # Update Account
    current_pnl = 0
    if pos != 0:
        if pos > 0: current_pnl = pos * (price - entry)
        else: current_pnl = abs(pos) * (entry - price)
        
    paper_account["balance"] = bal
    paper_account["position"] = pos
    paper_account["entry_price"] = entry
    paper_account["portfolio_value"] = bal + current_pnl

@app.get("/reset")
async def reset():
    global paper_account
    paper_account = {
        "balance": INITIAL_BALANCE,
        "position": 0.0,
        "entry_price": 0.0,
        "portfolio_value": INITIAL_BALANCE
    }
    return {"message": "Portfolio Reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)