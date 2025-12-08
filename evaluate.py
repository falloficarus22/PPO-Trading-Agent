import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
from data_loader import DataLoader
from src.environment import TradingEnvironment
from src.agent import PPOAgent

def evaluate_model(model_path, num_episodes=10, render=True):
    """
    Evaluate a trained PPO model on the trading environment.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of evaluation episodes
        render: Whether to plot results
    """
    print("=" * 50)
    print('Model Evaluation')
    print("=" * 50)
    
    # Load data
    print("\n[1/3] Loading Data...")
    loader = DataLoader(EXCHANGE_ID, SYMBOL, TIMEFRAME)
    df = loader.load_data()
    
    if df is None:
        print('No data found. Fetching from exchange...')
        df = loader.fetch_data(days=HISTORICAL_DAYS, save=True)
    
    print(f"Loaded {len(df)} candles")
    
    # Create environment
    print("\n[2/3] Creating Environment...")
    env = TradingEnvironment(
        df=df,
        initial_balance=INITIAL_BALANCE,
        trading_fee=TRADING_FEE,
        window_size=WINDOW_SIZE
    )
    
    # Create and load agent
    print("\n[3/3] Loading Model...")
    input_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    
    agent = PPOAgent(
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
    
    agent.load(model_path)
    agent.policy.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_path}\n")
    
    # Run evaluation episodes
    all_episode_rewards = []
    all_portfolio_values = []
    all_trades = []
    
    action_names = ['HOLD', 'BUY', 'SELL']
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        portfolio_values = [info['balance']]
        episode_trades = []
        steps = 0
        
        while True:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            portfolio_values.append(info.get('portfolio_value', info['balance']))
            
            if info.get('trades') and len(info['trades']) > len(episode_trades):
                episode_trades.extend(info['trades'][len(episode_trades):])
            
            state = next_state
            steps += 1
            
            if done or steps >= 1000:
                break
        
        all_episode_rewards.append(episode_reward)
        all_portfolio_values.append(portfolio_values)
        all_trades.append(episode_trades)
        
        final_value = info.get('portfolio_value', info['balance'])
        return_pct = ((final_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Final Portfolio Value: ${final_value:.2f}")
        print(f"  Return: {return_pct:.2f}%")
        print(f"  Total Reward: {episode_reward:.4f}")
        print(f"  Number of Trades: {len(episode_trades)}")
        print(f"  Steps: {steps}\n")
    
    # Calculate statistics
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    
    final_values = [pv[-1] for pv in all_portfolio_values]
    avg_final_value = np.mean(final_values)
    avg_return = ((avg_final_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    print("=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Average Final Portfolio Value: ${avg_final_value:.2f}")
    print(f"Average Return: {avg_return:.2f}%")
    print(f"Average Episode Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"Total Trades across all episodes: {sum(len(t) for t in all_trades)}")
    
    # Plot results
    if render and len(all_portfolio_values) > 0:
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio values for each episode
        plt.subplot(2, 1, 1)
        for i, pv in enumerate(all_portfolio_values):
            plt.plot(pv, alpha=0.5, label=f'Episode {i+1}')
        
        # Plot average
        max_len = max(len(pv) for pv in all_portfolio_values)
        avg_pv = []
        for step in range(max_len):
            step_values = [pv[step] if step < len(pv) else pv[-1] for pv in all_portfolio_values]
            avg_pv.append(np.mean(step_values))
        
        plt.plot(avg_pv, 'k-', linewidth=2, label='Average')
        plt.axhline(y=INITIAL_BALANCE, color='r', linestyle='--', label='Initial Balance')
        plt.xlabel('Step')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot episode rewards
        plt.subplot(2, 1, 2)
        plt.bar(range(1, num_episodes + 1), all_episode_rewards)
        plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Average: {avg_reward:.4f}')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = LOGS_DIR / 'evaluation_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
        plt.show()
    
    return {
        'episode_rewards': all_episode_rewards,
        'portfolio_values': all_portfolio_values,
        'trades': all_trades,
        'avg_reward': avg_reward,
        'avg_return': avg_return,
        'avg_final_value': avg_final_value
    }

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_path> [num_episodes]")
        print("Example: python evaluate.py models/ppo_final.pth 10")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    evaluate_model(model_path, num_episodes=num_episodes)

