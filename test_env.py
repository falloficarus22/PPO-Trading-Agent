import pandas as pd
from src.environment import TradingEnvironment
from data_loader import DataLoader
from config import *

def run_test():
    print('---Starting Environment Test---')

    print('Loadintg Data...')
    loader = DataLoader(EXCHANGE_ID, SYMBOL, TIMEFRAME)
    df = loader.load_data()

    if df is None:
        print('No data found, fetching from exchange...')
        df = loader.fetch_data(days = 30, save = True)

    print(f"Creating environment with {len(df)} candles...")
    env = TradingEnvironment(df, INITIAL_BALANCE, TRADING_FEE, WINDOW_SIZE)

    state, info = env.reset()
    print(f"Initial Balance: ${info['balance']}")
    print(f"Observation Shape: {state.shape}")

    print("\nRunning 10 random steps:")
    for i in range(10):
        action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)

        action_name = ['HOLD', 'BUY', 'SELL'][action]
        print(f"Step {i + 1}: Action = {action_name} | Price = ${env.df.loc[env.current_step, 'close']:.2f} | Reward = {reward:.4f}")

        if terminated or truncated:
            print('Episode ended early!')
            break

    print("\n--- Test Complete ---")

if __name__ == '__main__':
    run_test()