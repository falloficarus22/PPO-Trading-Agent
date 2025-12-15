import torch
import numpy as np
import pandas as pd
import ccxt
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

from config import *
from src.environment import TradingEnvironment
from src.agent import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPO Live Trading System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
REFRESH_RATE = 2.0
EXCHANGE_LIMIT = 500
TRADES_LOG_FILE = LOGS_DIR / 'live_trades.json'
PORTFOLIO_LOG_FILE = LOGS_DIR / 'portfolio_history.json'

# --- GLOBAL STATE ---
exchange = None
agent = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced Paper Trading State
paper_account = {
    "balance": INITIAL_BALANCE,
    "position": 0.0,
    "entry_price": 0.0,
    "portfolio_value": INITIAL_BALANCE,
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_pnl": 0.0,
    "max_portfolio_value": INITIAL_BALANCE,
    "min_portfolio_value": INITIAL_BALANCE,
    "trades_history": [],
    "start_time": None
}

# Data buffers
live_df_buffer = []
portfolio_history = []
last_action = "HOLD"
step_counter = 0

class TradeRecord(BaseModel):
    timestamp: datetime
    action: str
    price: float
    position: float
    balance: float
    pnl: float
    portfolio_value: float

@app.on_event("startup")
async def startup_event():
    global agent, device, exchange
    logger.info(f"Initializing Live Trading System on {device}...")
    
    try:
        # Initialize exchange
        # Initialize exchange dynamically based on config
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        logger.info("Exchange initialized successfully")
        
        # Load Agent
        OBS_DIM = WINDOW_SIZE * 12
        ACTION_DIM = 3
        
        agent = PPOAgent(OBS_DIM, ACTION_DIM, device=device)
        
        model_path = MODELS_DIR / 'ppo_final.pth'
        if model_path.exists():
            agent.load(model_path)
            agent.policy.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"No model found at {model_path}. Using untrained agent.")
        
        # Load historical data if exists
        load_portfolio_history()
        
        paper_account["start_time"] = datetime.now()
        logger.info("System Ready for Live Trading")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/")
async def read_root():
    """Serve the main dashboard"""
    return FileResponse("dashboard.html")

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": agent is not None,
        "exchange_connected": exchange is not None,
        "uptime": str(datetime.now() - paper_account["start_time"]) if paper_account["start_time"] else "0"
    }

@app.get("/history")
async def get_history():
    """Get historical market data for the chart"""
    try:
        if exchange is None:
             raise HTTPException(status_code=503, detail="Exchange not initialized")
             
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=EXCHANGE_LIMIT)
        
        # Format for lightweight charts
        data = []
        for candle in ohlcv:
            # timestamp, open, high, low, close, volume
            data.append({
                "time": int(candle[0] / 1000), # ms to s
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4]
            })
        return data
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/step")
async def step():
    """Execute a single trading step with live market data"""
    global live_df_buffer, portfolio_history, last_action, step_counter
    
    try:
        # 1. Fetch live market data
        logger.info("Fetching live market data...")
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=EXCHANGE_LIMIT)
        
        if not ohlcv:
            raise HTTPException(status_code=500, detail="No market data received")
        
        # 2. Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        current_price = float(df.iloc[-1]['close'])
        logger.info(f"Current BTC/USDT price: ${current_price:.2f}")
        
        # 3. Create environment with current data
        try:
            temp_env = TradingEnvironment(df, window_size=WINDOW_SIZE)
        except Exception as e:
            logger.error(f"Environment creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Environment error: {str(e)}")
        
        # 4. Get observation for latest window
        temp_env.current_step = len(temp_env.df) - 1
        temp_env.balance = paper_account["balance"]
        temp_env.position = paper_account["position"]
        
        obs = temp_env._get_observation()
        
        # 5. Get agent's action
        action = agent.get_action(obs, training=False)
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_str = action_map.get(action, 'HOLD')
        
        logger.info(f"Agent decision: {action_str}")
        
        # 6. Execute paper trade
        trade_result = execute_paper_trade(action, current_price)
        last_action = action_str
        step_counter += 1
        
        # 7. Update portfolio history
        portfolio_entry = {
            "time": int(datetime.now().timestamp()),
            "portfolio_value": float(paper_account["portfolio_value"]),
            "balance": float(paper_account["balance"]),
            "position": float(paper_account["position"]),
            "price": current_price
        }
        portfolio_history.append(portfolio_entry)
        
        # Keep only last 1000 entries
        if len(portfolio_history) > 1000:
            portfolio_history = portfolio_history[-1000:]
        
        # 8. Save to disk periodically
        if step_counter % 10 == 0:
            save_portfolio_history()
        
        # 9. Prepare candle data
        candle = {
            "time": int(df.iloc[-1]['timestamp'].timestamp()),
            "open": float(df.iloc[-1]['open']),
            "high": float(df.iloc[-1]['high']),
            "low": float(df.iloc[-1]['low']),
            "close": float(df.iloc[-1]['close']),
        }
        
        # 10. Calculate statistics
        stats = calculate_statistics()
        
        return {
            "step": step_counter,
            "candle": candle,
            "action": action_str,
            "balance": float(paper_account["balance"]),
            "position": float(paper_account["position"]),
            "portfolio_value": float(paper_account["portfolio_value"]),
            "entry_price": float(paper_account["entry_price"]),
            "mode": "LIVE",
            "statistics": stats,
            "trade_executed": trade_result
        }
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=503, detail=f"Exchange connection error: {str(e)}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error: {e}")
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def execute_paper_trade(action: int, price: float) -> bool:
    """Execute a paper trade and update account state"""
    global paper_account
    
    bal = paper_account["balance"]
    pos = paper_account["position"]
    entry = paper_account["entry_price"]
    fee_rate = TRADING_FEE
    
    trade_executed = False
    pnl = 0.0
    
    # 0=HOLD, 1=BUY, 2=SELL
    if action == 1:  # BUY
        if pos == 0:
            # Open Long
            pos = bal / price
            fee = bal * fee_rate
            bal -= fee
            entry = price
            trade_executed = True
            logger.info(f"OPENED LONG: {pos:.6f} BTC @ ${price:.2f}")
            
        elif pos < 0:
            # Close Short, Open Long
            pnl = abs(pos) * (entry - price)
            fee = abs(pos) * price * fee_rate
            bal += pnl - fee
            
            if pnl > 0:
                paper_account["winning_trades"] += 1
            else:
                paper_account["losing_trades"] += 1
            
            paper_account["total_trades"] += 1
            paper_account["total_pnl"] += pnl
            
            log_trade("CLOSE_SHORT", price, pos, pnl, fee)
            
            # Open Long
            pos = bal / price
            fee = bal * fee_rate
            bal -= fee
            entry = price
            trade_executed = True
            logger.info(f"CLOSED SHORT (PnL: ${pnl:.2f}), OPENED LONG @ ${price:.2f}")
            
    elif action == 2:  # SELL
        if pos == 0:
            # Open Short
            pos = -(bal / price)
            fee = abs(pos) * price * fee_rate
            bal -= fee
            entry = price
            trade_executed = True
            logger.info(f"OPENED SHORT: {abs(pos):.6f} BTC @ ${price:.2f}")
            
        elif pos > 0:
            # Close Long, Open Short
            pnl = pos * (price - entry)
            fee = pos * price * fee_rate
            bal += pnl - fee
            
            if pnl > 0:
                paper_account["winning_trades"] += 1
            else:
                paper_account["losing_trades"] += 1
            
            paper_account["total_trades"] += 1
            paper_account["total_pnl"] += pnl
            
            log_trade("CLOSE_LONG", price, pos, pnl, fee)
            
            # Open Short
            pos = -(bal / price)
            fee = abs(pos) * price * fee_rate
            bal -= fee
            entry = price
            trade_executed = True
            logger.info(f"CLOSED LONG (PnL: ${pnl:.2f}), OPENED SHORT @ ${price:.2f}")
    
    # Calculate current portfolio value
    current_pnl = 0
    if pos != 0:
        if pos > 0:
            current_pnl = pos * (price - entry)
        else:
            current_pnl = abs(pos) * (entry - price)
    
    portfolio_value = bal + current_pnl
    
    # Update account
    paper_account["balance"] = bal
    paper_account["position"] = pos
    paper_account["entry_price"] = entry
    paper_account["portfolio_value"] = portfolio_value
    
    # Update max/min portfolio value
    if portfolio_value > paper_account["max_portfolio_value"]:
        paper_account["max_portfolio_value"] = portfolio_value
    if portfolio_value < paper_account["min_portfolio_value"]:
        paper_account["min_portfolio_value"] = portfolio_value
    
    return trade_executed

def log_trade(action: str, price: float, position: float, pnl: float, fee: float):
    """Log trade to history"""
    trade = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "price": price,
        "position": position,
        "pnl": pnl,
        "fee": fee,
        "balance": paper_account["balance"],
        "portfolio_value": paper_account["portfolio_value"]
    }
    
    paper_account["trades_history"].append(trade)
    
    # Save to file
    try:
        with open(TRADES_LOG_FILE, 'w') as f:
            json.dump(paper_account["trades_history"], f, indent=2)
    except Exception as e:
        logger.error(f"Error saving trades: {e}")

def calculate_statistics() -> Dict:
    """Calculate trading statistics"""
    total_return = ((paper_account["portfolio_value"] - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    win_rate = 0
    if paper_account["total_trades"] > 0:
        win_rate = (paper_account["winning_trades"] / paper_account["total_trades"]) * 100
    
    max_drawdown = 0
    if paper_account["max_portfolio_value"] > 0:
        max_drawdown = ((paper_account["max_portfolio_value"] - paper_account["min_portfolio_value"]) 
                       / paper_account["max_portfolio_value"]) * 100
    
    return {
        "total_return": round(total_return, 2),
        "total_trades": paper_account["total_trades"],
        "winning_trades": paper_account["winning_trades"],
        "losing_trades": paper_account["losing_trades"],
        "win_rate": round(win_rate, 2),
        "total_pnl": round(paper_account["total_pnl"], 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": 0  # Placeholder for now
    }

def save_portfolio_history():
    """Save portfolio history to disk"""
    try:
        with open(PORTFOLIO_LOG_FILE, 'w') as f:
            json.dump(portfolio_history, f, indent=2)
        logger.info("Portfolio history saved")
    except Exception as e:
        logger.error(f"Error saving portfolio history: {e}")

def load_portfolio_history():
    """Load portfolio history from disk"""
    global portfolio_history
    try:
        if PORTFOLIO_LOG_FILE.exists():
            with open(PORTFOLIO_LOG_FILE, 'r') as f:
                portfolio_history = json.load(f)
            logger.info(f"Loaded {len(portfolio_history)} portfolio history entries")
    except Exception as e:
        logger.error(f"Error loading portfolio history: {e}")
        portfolio_history = []

@app.get("/portfolio_history")
async def get_portfolio_history():
    """Get complete portfolio history"""
    return {"history": portfolio_history}

@app.get("/trades_history")
async def get_trades_history():
    """Get all executed trades"""
    return {"trades": paper_account["trades_history"]}

@app.get("/statistics")
async def get_statistics():
    """Get detailed trading statistics"""
    stats = calculate_statistics()
    stats.update({
        "current_balance": paper_account["balance"],
        "current_position": paper_account["position"],
        "portfolio_value": paper_account["portfolio_value"],
        "max_portfolio_value": paper_account["max_portfolio_value"],
        "min_portfolio_value": paper_account["min_portfolio_value"],
        "uptime": str(datetime.now() - paper_account["start_time"]) if paper_account["start_time"] else "0"
    })
    return stats

@app.post("/reset")
async def reset():
    """Reset the trading system"""
    global paper_account, portfolio_history, step_counter, last_action
    
    paper_account = {
        "balance": INITIAL_BALANCE,
        "position": 0.0,
        "entry_price": 0.0,
        "portfolio_value": INITIAL_BALANCE,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "max_portfolio_value": INITIAL_BALANCE,
        "min_portfolio_value": INITIAL_BALANCE,
        "trades_history": [],
        "start_time": datetime.now()
    }
    
    portfolio_history = []
    step_counter = 0
    last_action = "HOLD"
    
    # Clear log files
    for log_file in [TRADES_LOG_FILE, PORTFOLIO_LOG_FILE]:
        if log_file.exists():
            log_file.unlink()
    
    logger.info("System reset completed")
    return {"message": "Portfolio and system reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)