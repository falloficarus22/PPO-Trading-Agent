import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from config import DATA_DIR
from utils import setup_logger

logger = setup_logger('data_loader')

class DataLoader:
    def __init__(self, exchange_id = 'binance', symbol = 'BTC/USDT', timeframe = '15m'):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})

    def fetch_data(self, days = 30, save = True):
        logger.info(f'Fetching {days} days of {self.timeframe} data for {self.symbol}...')
        since = self.exchange.parse8601((datetime.now() - timedelta(days = days)).isoformat())

        all_ohlcv = []
        limit = 1000

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since = since, limit = limit)

                if len(ohlcv) == 0:
                    break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                logger.info(f"fetched {len(all_ohlcv)} candles so far ...")

                if since >= self.exchange.milliseconds():
                    break

            except Exception as e:
                logger.error(f"Error in fetch loop: {e}")
                time.sleep(5)
                continue

        if len(all_ohlcv) == 0:
            logger.warning('No data fetched!')
            return None

        df = pd.DataFrame(all_ohlcv, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit = 'ms')

        logger.info(f"Successfully fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        if save:
            self.save_data(df)

        return df

    def save_data(self, df):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        filename = f"{self.exchange_id}_{self.symbol.replace('/', '_')}_{self.timeframe}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        df.to_csv(filepath, index = False)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename = None):
        if filename is None:
            filename = f"{self.exchange_id}_{self.symbol.replace('/', '_')}_{self.timeframe}.csv"
        
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
            
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} candles from {filepath}")
        return df