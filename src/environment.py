import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class TradingEnvironment(gym.Env):
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance  = 1000, trading_fee = 0.001, window_size = 50):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop = True)
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size

        self.smart_money_concepts()
        self.action_space = spaces.Discrete(3)

        obs_shape = (window_size, 12)
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = obs_shape,
            dtype = np.float32
        )

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.trades = []

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
        self.df.fillna(0, inplace = True)

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        window_data = self.df.iloc[start:end]

        close_price = self.df.loc[self.current_step, 'close']

        obs = []
        for _, row in window_data.iterrows():
            features = [
                # Price data features
                row['open'] / close_price,
                row['high'] / close_price,
                row['low'] / close_price,
                row['close'] / close_price,
                row['volume'] / (self.df['volume'].mean() + 1e-8),
                 
                # Smart money concept features

                # Distance to OBs
                (row['close'] - row['ob_bull_top']) / close_price if row['is_bull_ob'] else 0,
                (row['ob_bear_bottom'] - row['close']) / close_price if row['is_bear_ob'] else 0,

                # Are we still inside an OB
                1.0 if (row['low'] <= row['ob_bull_top']) and (row['high'] >= row['ob_bull_bottom']) else 0.0,
                1.0 if (row['low'] >= row['ob_bear_bottom']) and (row['high'] <= row['ob_bear_top']) else 0.0,

                # Contextual features
                row['volatility'] / close_price,
                self.balance / self.initial_balance,
                float(self.position)
            ]
            obs.append(features)

        while len(obs) < self.window_size:
            obs.insert(0, [0.0] * 12)

        return np.array(obs, dtype = np.float32)

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

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