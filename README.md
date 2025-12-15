  # PPO Trading Agent

<div align="center">

A sophisticated deep reinforcement learning trading agent that uses Proximal Policy Optimization (PPO) to autonomously trade cryptocurrency. The agent learns from historical price data and smart money concepts to make intelligent trading decisions.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Features

- **PPO Algorithm**: Implements the Proximal Policy Optimization algorithm for stable and efficient training
- **Smart Money Concepts**: Advanced technical analysis including:
  - Order blocks (supply/demand zones)
  - Liquidity zones
  - Break of Structure (BOS)
  - Change of Character (CHOCH)
- **Realistic Trading Environment**: Gymnasium-compatible environment with:
  - Realistic trading fees
  - Position management (long/short)
  - Portfolio tracking
  - Risk management
- **Data Integration**: Seamless data fetching from 100+ cryptocurrency exchanges via CCXT
- **Comprehensive Evaluation**: Built-in evaluation tools with detailed metrics and visualizations
- **Highly Configurable**: Easy-to-modify configuration for all hyperparameters and trading parameters
- **Live Trading Dashboard**: Real-time web dashboard for monitoring live trading performance with:
  - Real-time price and portfolio value charts
  - Trade execution tracking
  - Performance metrics
  - WebSocket-based live updates

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ppo-trading-agent.git
cd ppo-trading-agent
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

For GPU support (optional):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Test the Environment

Verify everything is set up correctly:

```bash
python test_env.py
```

Expected output:
```
---Starting Environment Test---
Loading Data...
Creating environment with X candles...
Initial Balance: $1000.00
Observation Shape: (50, 12)
Running 10 random steps:
...
```

### 2. Train a Model

Start training your first agent:

```bash
python main.py
```

The training process will:
- Automatically fetch historical data if not present
- Initialize the PPO agent
- Train for the configured number of episodes
- Save model checkpoints periodically

### 3. Evaluate a Trained Model

Evaluate your trained model:

```bash
python evaluate.py models/ppo_final.pth 10
```

This generates performance metrics and visualization plots.

## Configuration

All configuration is centralized in `config.py`. Here are the key parameters:

### Trading Parameters

```python
INITIAL_BALANCE = 1000      # Starting capital in USD
TRADING_FEE = 0.001         # Trading fee (0.1%)
WINDOW_SIZE = 50            # Observation window size
```

### PPO Hyperparameters

```python
LEARNING_RATE = 3e-4        # Adam optimizer learning rate
GAMMA = 0.99                # Discount factor for future rewards
GAE_LAMBDA = 0.95           # GAE lambda parameter
CLIP_EPSILON = 0.2          # PPO clipping parameter
ENTROPY_COEFF = 0.01        # Entropy bonus coefficient
VALUE_LOSS_COEFF = 0.5      # Value function loss coefficient
PPO_EPOCHS = 10             # Number of PPO update epochs
BATCH_SIZE = 64             # Training batch size
```

### Training Parameters

```python
NUM_EPISODES = 1000         # Total training episodes
SAVE_FREQUENCY = 50         # Model save interval
```

### Data Parameters

```python
EXCHANGE_ID = 'binance'     # Exchange name (ccxt supported)
SYMBOL = 'BTC/USDT'         # Trading pair
TIMEFRAME = '15m'           # Candle timeframe
HISTORICAL_DAYS = 90        # Days of historical data
```

## Usage

### Training

```bash
python main.py
```

During training, you'll see progress updates every 10 episodes:
```
Episode 10/1000 | Avg Reward: -0.0234 | Avg Length: 234.5 | Balance: $987.45
Episode 20/1000 | Avg Reward: 0.0145 | Avg Length: 256.2 | Balance: $1023.12
...
```

### Evaluation

Evaluate a specific model:

```bash
python evaluate.py models/ppo_ep500.pth 20
```

Arguments:
- `model_path`: Path to the saved model (.pth file)
- `num_episodes`: Number of evaluation episodes (optional, default: 10)

Output includes:
- Average portfolio value
- Return percentage
- Number of trades
- Performance visualizations

### Custom Data

To use your own data:

1. Place a CSV file in the `data/` directory with columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
2. Modify `data_loader.py` to load your file
3. Or update `EXCHANGE_ID`, `SYMBOL`, and `TIMEFRAME` in `config.py`

## Architecture

### Agent Architecture

The PPO agent uses an **Actor-Critic** architecture:

```
Input (Observation) → Shared Network (2-layer MLP) → ┬→ Actor Head → Action Probabilities
                                                     └→ Critic Head → State Value
```

- **Shared Network**: 256 → 128 hidden units with ReLU activations
- **Actor Head**: Outputs probability distribution over actions (HOLD, BUY, SELL)
- **Critic Head**: Estimates state value for advantage calculation

### Trading Environment

The environment implements:

- **Action Space**: Discrete 3 actions (HOLD, BUY, SELL)
- **Observation Space**: `(window_size, 12)` feature matrix containing:
  1. Normalized OHLCV data (5 features)
  2. Smart money concept features (5 features)
  3. Portfolio state (2 features: balance ratio, position)

- **Reward Function**: Portfolio value change normalized by initial balance
- **Termination**: Episode ends if portfolio drops below 10% of initial balance

### Smart Money Concepts

The environment calculates several advanced trading concepts:

1. **Swing Highs/Lows**: Identifies local price extremes
2. **Liquidity Zones**: Areas where stop losses cluster
3. **Order Blocks**: Supply/demand zones based on BOS/CHOCH
4. **Break of Structure (BOS)**: Trend continuation signals
5. **Change of Character (CHOCH)**: Trend reversal signals

## Project Structure

```
ppo-trading-agent/
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── agent.py             # PPO agent implementation
│   ├── environment.py       # Trading environment (Gymnasium)
│   └── trainer.py           # Training loop and utilities
│
├── data/                    # Historical price data (CSV files)
├── models/                  # Saved model checkpoints (.pth files)
├── logs/                    # Training logs and evaluation plots
│
├── config.py                # Centralized configuration
├── data_loader.py           # Data fetching and loading utilities
├── evaluate.py              # Model evaluation script
├── main.py                  # Main training script
├── test_env.py              # Environment testing script
├── utils.py                 # Utility functions (logging, etc.)
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Examples

### Example 1: Train on Different Cryptocurrency Pair

Edit `config.py`:
```python
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
HISTORICAL_DAYS = 180
```

Then train:
```bash
python main.py
```

### Example 2: Custom Hyperparameters for Faster Training

Edit `config.py`:
```python
NUM_EPISODES = 500
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
```

### Example 3: Evaluate Multiple Models

```bash
# Evaluate different checkpoints
python evaluate.py models/ppo_ep100.pth 20
python evaluate.py models/ppo_ep200.pth 20
python evaluate.py models/ppo_final.pth 20
```

## Troubleshooting

### Issue: "No data found"

**Solution**: The script will automatically fetch data from the exchange. Ensure you have:
- Internet connectivity
- Valid `EXCHANGE_ID` in config
- Exchange supports the specified symbol/timeframe

### Issue: CUDA/GPU not working

**Solution**: 
1. Install CUDA-compatible PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Modify `src/agent.py`:
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

### Issue: Out of Memory

**Solution**: Reduce one or more of these in `config.py`:
- `BATCH_SIZE` (try 32 or 16)
- `WINDOW_SIZE` (try 30 or 40)
- `NUM_EPISODES` (train in smaller batches)

### Issue: Training is too slow

**Solutions**:
- Use GPU (see above)
- Reduce `NUM_EPISODES` and `PPO_EPOCHS`
- Use smaller `WINDOW_SIZE` or `BATCH_SIZE`

### Issue: Poor Performance

**Solutions**:
- Train for more episodes (`NUM_EPISODES`)
- Tune hyperparameters (learning rate, gamma)
- Try different timeframes or symbols
- Increase `WINDOW_SIZE` for more context

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Test thoroughly** (`python test_env.py`)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Areas for Contribution

- Additional technical indicators
- New reward functions
- Support for multiple assets
- Backtesting improvements
- Documentation improvements
- Bug fixes

## Disclaimer

**This project is for educational and research purposes only.** 

- Cryptocurrency trading involves substantial risk
- Past performance does not guarantee future results
- Always do your own research before making trading decisions
- Never invest more than you can afford to lose
- This agent is not financial advice

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch** - Deep learning framework
- **Gymnasium** - RL environment standard
- **CCXT** - Cryptocurrency exchange library
- **OpenAI** - PPO algorithm inspiration
- The open-source community for inspiration and tools

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [CCXT Documentation](https://docs.ccxt.com/)

## Future Improvements

- [ ] Support for multiple assets/portfolio trading
- [ ] Additional technical indicators (RSI, MACD, etc.)
- [ ] Live trading integration (paper trading first!)
- [ ] TensorBoard integration for better visualization
- [ ] Distributed training support
- [ ] Model ensembling
- [ ] Risk management modules

---

<div align="center">

**If you find this project useful, please consider giving it a star!**

Made with dedication by the open-source community

</div>
