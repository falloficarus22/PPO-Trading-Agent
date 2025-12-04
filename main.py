import numpy as np
from config import *
from data_loader import DataLoader
from src.environment import TradingEnvironment
from src.agent import PPOAgent
from src.trainer import Trainer

def main():
    print("=" * 50)
    print('PPO Trader')
    print("=" * 50)

    print("\n[1/4] Loading Data...")

    loader = DataLoader(EXCHANGE_ID, SYMBOL, TIMEFRAME)
    df = loader.load_data()

    if df is None:
        print('No data found. Fetching from exhange...')
        df = loader.fetch_data(days = HISTORICAL_DAYS, save = True)

    print(f"Loaded {len(df)} candles")

    print("\n[2/4] Creating Environment...")
    env = TradingEnvironment(
        df = df,
        initial_balance = INITIAL_BALANCE,
        trading_fee = TRADING_FEE,
        window_size = WINDOW_SIZE
    )
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n} actions")

    print("\n[3/4] Creating PPO Agent...")
    input_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    agent = PPOAgent(
        input_dim = input_dim,
        action_dim = action_dim,
        lr = LEARNING_RATE,
        gamma = GAMMA,
        gae_lambda = GAE_LAMBDA,
        clip_epsilon = CLIP_EPSILON,
        entropy_coeff = ENTROPY_COEFF,
        value_loss_coeff = VALUE_LOSS_COEFF,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE
    )
    print(f"Network parameters: {sum(p.numel() for p in agent.policy.parameters())}")
    print("\n[4/4] Training...")
    trainer = Trainer(env, agent, save_dir = str(MODELS_DIR))

    rewards, lengths = trainer.train(
        num_episodes = NUM_EPISODES,
        max_steps = 1000,
        save_interval = SAVE_FREQUENCY
    )

    print('\nTraining completed.')
    print(f"Final avg reward (last 100): {np.mean(rewards[-100:]):.4f}")
    
if __name__ == '__main__':
    main()