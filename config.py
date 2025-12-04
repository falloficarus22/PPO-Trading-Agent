import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
LOGS_DIR = ROOT_DIR / 'logs'

for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok = True, parents = True)

EXCHANGE_ID = 'binance'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'

INITIAL_BALANCE = 1000
TRADING_FEE = 0.001
WINDOW_SIZE = 50

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
PPO_EPOCHS = 10
BATCH_SIZE = 64

NUM_EPISODES = 1000
UPDATE_FREQUENCY = 2048
SAVE_FREQUENCY = 50
EVAL_FREQUENCY = 10

HIDDEN_DIM = 256
INPUT_DIM = 50 * 12
ACTION_DIM = 3

HISTORICAL_DAYS = 90