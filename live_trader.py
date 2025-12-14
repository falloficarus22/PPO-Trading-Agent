import numpy as np
import pandas as pd
import ccxt
import time
import threading
from datetime import datetime
from collections import deque
from pathlib import Path
import json
import torch

from config import *
from data_loader import DataLoader
from src.environment import TradingEnvironment
from src.agent import PPOAgent

class LiveTrader:
    def __init__(self, model_path, initial_balance=None):
        self.model_path = model_path
        self.initial_balance = initial_balance or INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trading_fee = TRADING_FEE
        self.window_size = WINDOW_SIZE
        
        # Load agent
        self.loader = DataLoader(EXCHANGE_ID, SYMBOL, TIMEFRAME)
        input_dim = WINDOW_SIZE * 12
        action_dim = 3
        
        self.agent = PPOAgent(
            input_dim=input_dim,
            action_dim=action_dim,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_epsilon=CLIP_EPSILON,
            entropy_coeff=ENTROPY_COEFF,
            value_loss_coeff=VALUE_LOSS_COEFF,
            epochs=PPO_EPOCHS,
            batch_size=BATCH_SIZE
        )
        self.agent.load(model_path)
        self.agent.policy.eval()
        
        # Data storage
        self.price_history = deque(maxlen=1000)
        self.trades_history = []
        self.portfolio_values = []
        self.actions_history = []
        self.timestamps = deque(maxlen=1000)
        
        # Get initial data
        self.df = self.loader.load_data()
        if self.df is None or len(self.df) < WINDOW_SIZE:
            print("Fetching initial historical data...")
            self.df = self.loader.fetch_data(days=7, save=False)
        
        # Prepare environment for feature calculation
        self.env = TradingEnvironment(
            df=self.df.copy(),
            initial_balance=self.initial_balance,
            trading_fee=self.trading_fee,
            window_size=self.window_size
        )
        
        # Initialize price buffer
        if len(self.df) > 0:
            latest_candle = self.df.iloc[-1]
            self.price_history.append({
                'timestamp': latest_candle.get('timestamp', datetime.now()),
                'price': latest_candle['close'],
                'high': latest_candle['high'],
                'low': latest_candle['low'],
                'open': latest_candle['open'],
                'volume': latest_candle['volume']
            })
        
        self.running = False
        self.lock = threading.Lock()
        
    def get_latest_candle(self):
        """Fetch the latest candle from exchange"""
        try:
            exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1)
            if len(ohlcv) > 0:
                candle = ohlcv[-1]
                return {
                    'timestamp': pd.to_datetime(candle[0], unit='ms'),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                }
        except Exception as e:
            print(f"Error fetching candle: {e}")
        return None
    
    def update_price_history(self, candle):
        """Update price history with new candle"""
        with self.lock:
            # Check if this is a new candle or update to current
            if len(self.price_history) == 0 or \
               candle['timestamp'] > self.price_history[-1]['timestamp']:
                self.price_history.append(candle)
                self.timestamps.append(candle['timestamp'])
            else:
                # Update last candle
                self.price_history[-1].update(candle)
                self.timestamps[-1] = candle['timestamp']
    
    def prepare_observation(self):
        """Prepare observation from recent price history"""
        if len(self.price_history) < self.window_size:
            return None
        
        # Create dataframe from recent history
        recent_data = list(self.price_history)[-self.window_size:]
        df_window = pd.DataFrame(recent_data)
        
        # Need to calculate smart money concepts, so we need more context
        # Append to main df for calculation
        df_extended = self.df.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df_extended.columns:
            df_extended['timestamp'] = pd.to_datetime(df_extended.index)
        
        # Add new data
        for candle in recent_data:
            candle_ts = pd.to_datetime(candle['timestamp'])
            if len(df_extended) == 0 or candle_ts > df_extended['timestamp'].max():
                new_row = pd.DataFrame([{
                    'timestamp': candle_ts,
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['price'],
                    'volume': candle['volume']
                }])
                df_extended = pd.concat([df_extended, new_row], ignore_index=True)
        
        # Calculate features using environment
        temp_env = TradingEnvironment(
            df=df_extended,
            initial_balance=self.initial_balance,
            trading_fee=self.trading_fee,
            window_size=self.window_size
        )
        
        # Set to latest step
        temp_env.current_step = len(df_extended) - 1
        temp_env.balance = self.balance
        temp_env.position = self.position
        temp_env.entry_price = self.entry_price
        
        obs = temp_env._get_observation()
        return obs
    
    def execute_action(self, action, current_price):
        """Execute trading action"""
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        trade_executed = False
        trade_info = None
        
        with self.lock:
            if action == 1:  # BUY
                if self.position == 0:
                    # Open long
                    position_size = self.balance / current_price
                    self.position = position_size
                    self.entry_price = current_price
                    fee = self.balance * self.trading_fee
                    self.balance -= fee
                    trade_executed = True
                    trade_info = {
                        'type': 'OPEN_LONG',
                        'price': current_price,
                        'size': position_size,
                        'fee': fee
                    }
                elif self.position < 0:
                    # Close short and open long
                    pnl = abs(self.position) * (self.entry_price - current_price)
                    fee_close = abs(self.position) * current_price * self.trading_fee
                    self.balance += pnl - fee_close
                    
                    trade_info = {
                        'type': 'CLOSE_SHORT',
                        'price': current_price,
                        'pnl': pnl,
                        'fee': fee_close
                    }
                    
                    position_size = self.balance / current_price
                    fee_open = self.balance * self.trading_fee
                    self.balance -= fee_open
                    self.position = position_size
                    self.entry_price = current_price
                    trade_executed = True
                    
            elif action == 2:  # SELL
                if self.position == 0:
                    # Open short
                    position_size = -(self.balance / current_price)
                    fee = abs(position_size) * current_price * self.trading_fee
                    self.balance -= fee
                    self.position = position_size
                    self.entry_price = current_price
                    trade_executed = True
                    trade_info = {
                        'type': 'OPEN_SHORT',
                        'price': current_price,
                        'size': abs(position_size),
                        'fee': fee
                    }
                elif self.position > 0:
                    # Close long and open short
                    pnl = self.position * (current_price - self.entry_price)
                    fee_close = self.position * current_price * self.trading_fee
                    self.balance += pnl - fee_close
                    
                    trade_info = {
                        'type': 'CLOSE_LONG',
                        'price': current_price,
                        'pnl': pnl,
                        'fee': fee_close
                    }
                    
                    position_size = -(self.balance / current_price)
                    fee_open = abs(position_size) * current_price * self.trading_fee
                    self.balance -= fee_open
                    self.position = position_size
                    self.entry_price = current_price
                    trade_executed = True
        
        return trade_executed, trade_info
    
    def get_portfolio_value(self, current_price):
        """Calculate current portfolio value"""
        if self.position > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price)
        elif self.position < 0:
            unrealized_pnl = abs(self.position) * (self.entry_price - current_price)
        else:
            unrealized_pnl = 0
        
        return self.balance + unrealized_pnl
    
    def run_step(self):
        """Run one trading step"""
        candle = self.get_latest_candle()
        if candle is None:
            return None
        
        self.update_price_history(candle)
        current_price = candle['price']
        
        obs = self.prepare_observation()
        if obs is None:
            return {
                'price': current_price,
                'timestamp': candle['timestamp'].isoformat() if hasattr(candle['timestamp'], 'isoformat') else str(candle['timestamp']),
                'status': 'warming_up'
            }
        
        # Get action from agent
        action = self.agent.get_action(obs, training=False)
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        
        # Execute action
        trade_executed, trade_info = self.execute_action(action, current_price)
        
        # Calculate portfolio metrics
        portfolio_value = self.get_portfolio_value(current_price)
        unrealized_pnl = portfolio_value - self.balance
        total_return = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        # Store history
        timestamp = candle['timestamp']
        if hasattr(timestamp, 'isoformat'):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)
        
        self.portfolio_values.append({
            'timestamp': timestamp_str,
            'value': portfolio_value,
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl
        })
        
        self.actions_history.append({
            'timestamp': timestamp_str,
            'action': action_name,
            'price': current_price
        })
        
        if trade_executed and trade_info:
            trade_record = {
                'timestamp': timestamp_str,
                **trade_info
            }
            self.trades_history.append(trade_record)
        
        return {
            'timestamp': timestamp_str,
            'price': current_price,
            'action': action_name,
            'action_id': int(action),
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price if self.position != 0 else None,
            'unrealized_pnl': unrealized_pnl,
            'total_return': total_return,
            'trade_executed': trade_executed,
            'trade_info': trade_info
        }
    
    def get_status(self):
        """Get current trading status"""
        if len(self.price_history) == 0:
            return {
                'status': 'initializing',
                'balance': self.balance,
                'position': 0
            }
        
        current_price = self.price_history[-1]['price']
        portfolio_value = self.get_portfolio_value(current_price)
        
        return {
            'status': 'running' if self.running else 'stopped',
            'initial_balance': self.initial_balance,
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price if self.position != 0 else None,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'unrealized_pnl': portfolio_value - self.balance,
            'total_return': ((portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': len(self.trades_history),
            'price_history_length': len(self.price_history)
        }
    
    def get_history(self, limit=100):
        """Get trading history"""
        return {
            'portfolio_values': list(self.portfolio_values[-limit:]),
            'trades': list(self.trades_history[-limit:]),
            'actions': list(self.actions_history[-limit:]),
            'prices': [{'timestamp': str(ts) if hasattr(ts, 'isoformat') else str(ts), 
                       'price': p['price']} 
                      for ts, p in zip(list(self.timestamps)[-limit:], 
                                       list(self.price_history)[-limit:])]
        }

