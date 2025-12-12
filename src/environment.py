import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class TradingEnvironment(gym.Env):
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=1000, trading_fee=0.001, window_size=50):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size

        self.smart_money_concepts()
        self.precompute_features()  # <--- NEW: Calculate everything once

        self.action_space = spaces.Discrete(3)

        obs_shape = (window_size, 12)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        self.reset()

    def _identify_liquidity(self):
        # Swing highs and lows
        self.df['swing_high'] = (
            (self.df['high'] > self.df['high'].shift(1)) & 
            (self.df['high'] > self.df['high'].shift(2)) & 
            (self.df['high'] > self.df['high'].shift(3)) & 
            (self.df['high'] > self.df['high'].shift(-1)) & 
            (self.df['high'] > self.df['high'].shift(-2)) & 
            (self.df['high'] > self.df['high'].shift(-3))
        )

        self.df['swing_low'] = (
            (self.df['low'] < self.df['low'].shift(1)) & 
            (self.df['low'] < self.df['low'].shift(2)) & 
            (self.df['low'] < self.df['low'].shift(3)) & 
            (self.df['low'] < self.df['low'].shift(-1)) & 
            (self.df['low'] < self.df['low'].shift(-2)) & 
            (self.df['low'] < self.df['low'].shift(-3))
        )

        self.df['liquidity_high'] = self.df['high'].where(self.df['swing_high']).ffill()
        self.df['liquidity_high_low'] = self.df['low'].where(self.df['swing_high']).ffill()

        self.df['liquidity_low'] = self.df['low'].where(self.df['swing_low']).ffill()
        self.df['liquidity_low_high'] = self.df['high'].where(self.df['swing_low']).ffill()

    def smart_money_concepts(self):
        # Identify swing points first
        self._identify_liquidity()

        bull_break = (self.df['close'] > self.df['liquidity_high'].shift(1))
        bear_break = (self.df['close'] < self.df['liquidity_low'].shift(1))

        self.df['break_signal'] = np.nan
        self.df.loc[bull_break, 'break_signal'] = 1
        self.df.loc[bear_break, 'break_signal'] = -1

        # Identify trend flow
        self.df['last_break'] = self.df['break_signal'].shift(1).ffill()

        self.df['bull_bos'] = (self.df['break_signal'] == 1) & (self.df['last_break'] == 1)
        self.df['bear_bos'] = (self.df['break_signal'] == -1) & (self.df['last_break'] == -1)

        self.df['bull_choch'] = (self.df['break_signal'] == 1) & (self.df['last_break'] == -1)
        self.df['bear_choch'] = (self.df['break_signal'] == -1) & (self.df['last_break'] == 1)

        self.df['is_bull_ob'] = False
        self.df['is_bear_ob'] = False

        self.df.loc[self.df['bull_bos'] | self.df['bull_choch'], 'is_bull_ob'] = True
        self.df.loc[self.df['bear_bos'] | self.df['bear_choch'], 'is_bear_ob'] = True

        # Bullish order block
        self.df['ob_bull_top'] = self.df['liquidity_low_high'].where(self.df['is_bull_ob']).ffill()
        self.df['ob_bull_bottom'] = self.df['liquidity_low'].where(self.df['is_bull_ob']).ffill()

        # Bearish order block
        self.df['ob_bear_top'] = self.df['liquidity_high'].where(self.df['is_bear_ob']).ffill()
        self.df['ob_bear_bottom'] = self.df['liquidity_high_low'].where(self.df['is_bear_ob']).ffill()

        self.df['volatility'] = self.df['high'] - self.df['low']
        self.df.fillna(0, inplace=True)

    def precompute_features(self):
        """Precompute all static features to avoid loops during steps"""
        # 1. Price Data Features
        close = self.df['close']
        
        feat_open = self.df['open'] / close
        feat_high = self.df['high'] / close
        feat_low = self.df['low'] / close
        feat_close = self.df['close'] / close
        feat_vol = self.df['volume'] / (self.df['volume'].mean() + 1e-8)

        # 2. SMC Features
        # Distance to OBs
        # Need to handle the conditional 'if row[is_bull_ob]' vectorially
        # We can just compute the value for all rows, then zero out non-OB rows
        dist_bull = (self.df['close'] - self.df['ob_bull_top']) / close
        dist_bull = np.where(self.df['is_bull_ob'], dist_bull, 0)

        dist_bear = (self.df['ob_bear_bottom'] - self.df['close']) / close
        dist_bear = np.where(self.df['is_bear_ob'], dist_bear, 0)

        # Inside OB
        in_bull = (self.df['low'] <= self.df['ob_bull_top']) & (self.df['high'] >= self.df['ob_bull_bottom'])
        in_bear = (self.df['low'] >= self.df['ob_bear_bottom']) & (self.df['high'] <= self.df['ob_bear_top'])
        
        # 3. Contextual Features (minus balance/position which are dynamic)
        feat_vola = self.df['volatility'] / close

        # Stack static features
        # Shape: (N, 10) -> We will append the 2 dynamic features later
        self.static_features = np.column_stack([
            feat_open, feat_high, feat_low, feat_close, feat_vol,
            dist_bull, dist_bear,
            in_bull.astype(float), in_bear.astype(float),
            feat_vola
        ]).astype(np.float32)

    def _get_observation(self):
        # Fast slicing using the precomputed array
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        
        # Get static features for the window
        # If we are at the very start (step < window_size), we need to pad
        static_chunk = self.static_features[start:end]
        
        if len(static_chunk) < self.window_size:
            pad_len = self.window_size - len(static_chunk)
            # Pad with zeros
            padding = np.zeros((pad_len, 10), dtype=np.float32)
            static_chunk = np.vstack([padding, static_chunk])

        # Current dynamic features
        # We need to broadcast these scalar values to match the window size (50, 2)
        # Or, usually in RL, we just attach them to the current step or all steps.
        # Your previous loop attached them to *every* step in the window.
        dynamic_features = np.array([
            self.balance / self.initial_balance,
            float(self.position)
        ], dtype=np.float32)
        
        # Broadcast (50, 2)
        dynamic_chunk = np.tile(dynamic_features, (self.window_size, 1))

        # Combine (50, 10) + (50, 2) -> (50, 12)
        obs = np.concatenate([static_chunk, dynamic_chunk], axis=1)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = np.random.randint(self.window_size, len(self.df) - 100)
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.trades = []

        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'total_pnl': self.total_pnl,
            'trades': self.trades
        }

        return observation, info

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']

        if action == 0:  # HOLD
            pass
        elif action == 1:  # BUY
            if self.position == 0:
                # Open long position
                position_size = self.balance / current_price
                self.position = position_size
                self.entry_price = current_price
                fee = self.balance * self.trading_fee
                self.balance -= fee
            elif self.position < 0:
                # Close short position and open long
                pnl = abs(self.position) * (self.entry_price - current_price)
                fee_close = abs(self.position) * current_price * self.trading_fee
                self.balance += pnl - fee_close
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'close_short',
                    'pnl': pnl,
                    'fee': fee_close,
                    'balance': self.balance,
                    'position': self.position,
                    'entry_price': self.entry_price
                })
                
                # Open new long position
                position_size = self.balance / current_price
                fee_open = self.balance * self.trading_fee
                self.balance -= fee_open
                self.position = position_size
                self.entry_price = current_price
        else:  # SELL (action == 2)
            if self.position == 0:
                # Open short position
                position_size = -(self.balance / current_price)
                fee = abs(position_size) * current_price * self.trading_fee
                self.balance -= fee
                self.position = position_size
                self.entry_price = current_price
            elif self.position > 0:
                # Close long position and open short
                pnl = self.position * (current_price - self.entry_price)
                fee_close = self.position * current_price * self.trading_fee
                self.balance += pnl - fee_close
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'close_long',
                    'pnl': pnl,
                    'fee': fee_close,
                    'balance': self.balance,
                    'position': self.position,
                    'entry_price': self.entry_price
                })
                
                # Open new short position
                position_size = -(self.balance / current_price)
                fee_open = abs(position_size) * current_price * self.trading_fee
                self.balance -= fee_open
                self.position = position_size
                self.entry_price = current_price

        self.current_step += 1
        
        # Calculate unrealized PnL (check bounds first)
        if self.position != 0 and self.current_step < len(self.df):
            current_price_new = self.df.loc[self.current_step, 'close']
            if self.position > 0:
                unrealized_pnl = self.position * (current_price_new - self.entry_price)
            else:
                unrealized_pnl = abs(self.position) * (self.entry_price - current_price_new)
        else:
            unrealized_pnl = 0

        portfolio_value = self.balance + unrealized_pnl
        reward = (portfolio_value - self.initial_balance) / self.initial_balance

        terminated = False
        if portfolio_value <= self.initial_balance * 0.1:
            terminated = True
            reward = -10

        truncated = False
        if self.current_step >= len(self.df) - 1:
            truncated = True

        observation = self._get_observation()

        info = {
            'balance': self.balance,
            'position': self.position, 
            'entry_price': self.entry_price,
            'portfolio_value': portfolio_value,
            'total_pnl': self.total_pnl, 
            'unrealized_pnl': unrealized_pnl,
            'current_step': self.current_step,
            'trades': self.trades
        }

        return observation, reward, terminated, truncated, info