"""
Script to test technical indicators for market regime testing
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import talib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_regime_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TechnicalIndicatorsTester:
    """
    Class to test technical indicators for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the technical indicators tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/technical_indicators')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Technical indicator parameters
        self.ema_short_period = self.config.get('ema_short_period', 9)
        self.ema_medium_period = self.config.get('ema_medium_period', 21)
        self.ema_long_period = self.config.get('ema_long_period', 50)
        self.vwap_period = self.config.get('vwap_period', 14)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast_period = self.config.get('macd_fast_period', 12)
        self.macd_slow_period = self.config.get('macd_slow_period', 26)
        self.macd_signal_period = self.config.get('macd_signal_period', 9)
        self.bollinger_period = self.config.get('bollinger_period', 20)
        self.bollinger_std_dev = self.config.get('bollinger_std_dev', 2)
        self.atr_period = self.config.get('atr_period', 14)
        
        logger.info(f"Initialized TechnicalIndicatorsTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"Technical indicator parameters: ema_short_period={self.ema_short_period}, ema_medium_period={self.ema_medium_period}, ema_long_period={self.ema_long_period}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for technical indicators testing")
        
        # Try to load merged data first
        merged_data_path = os.path.join(self.data_dir, "merged_data.csv")
        if os.path.exists(merged_data_path):
            logger.info(f"Loading merged data from {merged_data_path}")
            df = pd.read_csv(merged_data_path)
            logger.info(f"Loaded merged data with {len(df)} rows")
            return df
        
        # If merged data doesn't exist, try to load individual processed files
        logger.info("Merged data not found, looking for individual processed files")
        processed_files = [f for f in os.listdir(self.data_dir) if f.startswith("processed_") and f.endswith(".csv")]
        
        if not processed_files:
            logger.error("No processed data files found")
            return None
        
        logger.info(f"Found {len(processed_files)} processed data files")
        dfs = []
        
        for file_name in processed_files:
            file_path = os.path.join(self.data_dir, file_name)
            logger.info(f"Loading data from {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_name}")
            except Exception as e:
                logger.error(f"Error loading {file_name}: {str(e)}")
        
        if not dfs:
            logger.error("No data loaded from processed files")
            return None
        
        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged {len(dfs)} dataframes with total {len(merged_df)} rows")
        
        return merged_df
    
    def prepare_price_data(self, df):
        """
        Prepare price data for technical indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Prepared price data
        """
        logger.info("Preparing price data for technical indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have price data
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # If Underlying_Price exists, use it as Close
        if 'Underlying_Price' in result_df.columns and 'Close' not in result_df.columns:
            logger.info("Using 'Underlying_Price' column as 'Close'")
            result_df['Close'] = result_df['Underlying_Price']
        
        # If Price exists, use it as Close
        if 'Price' in result_df.columns and 'Close' not in result_df.columns:
            logger.info("Using 'Price' column as 'Close'")
            result_df['Close'] = result_df['Price']
        
        # Check if we have Close price
        if 'Close' not in result_df.columns:
            logger.error("No Close price data available")
            return None
        
        # If we don't have Open, High, Low, create them from Close
        if 'Open' not in result_df.columns:
            logger.info("Creating 'Open' column from 'Close'")
            result_df['Open'] = result_df['Close'].shift(1)
            # For the first row, use Close as Open
            result_df.loc[0, 'Open'] = result_df.loc[0, 'Close']
        
        if 'High' not in result_df.columns:
            logger.info("Creating 'High' column from 'Close' and 'Open'")
            result_df['High'] = result_df[['Close', 'Open']].max(axis=1)
            # Add some randomness to High
            np.random.seed(42)  # For reproducibility
            result_df['High'] = result_df['High'] * (1 + np.random.uniform(0, 0.005, len(result_df)))
        
        if 'Low' not in result_df.columns:
            logger.info("Creating 'Low' column from 'Close' and 'Open'")
            result_df['Low'] = result_df[['Close', 'Open']].min(axis=1)
            # Add some randomness to Low
            np.random.seed(43)  # For reproducibility
            result_df['Low'] = result_df['Low'] * (1 - np.random.uniform(0, 0.005, len(result_df)))
        
        if 'Volume' not in result_df.columns:
            logger.info("Creating synthetic 'Volume' column")
            # Create synthetic volume data
            np.random.seed(44)  # For reproducibility
            result_df['Volume'] = np.random.randint(1000, 10000, len(result_df))
            # Make volume correlate with price changes
            result_df['Price_Change'] = result_df['Close'].pct_change().abs()
            result_df['Volume'] = result_df['Volume'] * (1 + 5 * result_df['Price_Change'])
            result_df['Volume'] = result_df['Volume'].fillna(result_df['Volume'].mean()).astype(int)
            result_df.drop('Price_Change', axis=1, inplace=True)
        
        # Ensure datetime is in datetime format
        if 'datetime' in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df['datetime']):
            try:
                result_df['datetime'] = pd.to_datetime(result_df['datetime'])
                logger.info("Converted datetime to datetime format")
            except Exception as e:
                logger.warning(f"Failed to convert datetime to datetime format: {str(e)}")
                
                # If conversion fails, try to create datetime from Date and Time columns
                if 'Date' in result_df.columns and 'Time' in result_df.columns:
                    try:
                        result_df['datetime'] = pd.to_datetime(result_df['Date'].astype(str) + ' ' + result_df['Time'].astype(str))
                        logger.info("Created datetime from Date and Time columns")
                    except Exception as e2:
                        logger.error(f"Failed to create datetime from Date and Time: {str(e2)}")
        
        # Sort by datetime
        if 'datetime' in result_df.columns:
            result_df.sort_values('datetime', inplace=True)
            logger.info("Sorted data by datetime")
        
        # Reset index
        result_df.reset_index(drop=True, inplace=True)
        
        logger.info("Prepared price data for technical indicators")
        
        return result_df
    
    def calculate_ema_indicators(self, df):
        """
        Calculate EMA indicators
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with EMA indicators
        """
        logger.info("Calculating EMA indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have Close price
        if 'Close' not in result_df.columns:
            logger.error("No Close price data available for EMA calculation")
            return result_df
        
        try:
            # Calculate EMAs
            result_df[f'EMA_{self.ema_short_period}'] = talib.EMA(result_df['Close'], timeperiod=self.ema_short_period)
            result_df[f'EMA_{self.ema_medium_period}'] = talib.EMA(result_df['Close'], timeperiod=self.ema_medium_period)
            result_df[f'EMA_{self.ema_long_period}'] = talib.EMA(result_df['Close'], timeperiod=self.ema_long_period)
            
            # Calculate EMA crossovers
            result_df['EMA_Short_Medium_Crossover'] = np.where(
                result_df[f'EMA_{self.ema_short_period}'] > result_df[f'EMA_{self.ema_medium_period}'], 1,
                np.where(result_df[f'EMA_{self.ema_short_period}'] < result_df[f'EMA_{self.ema_medium_period}'], -1, 0)
            )
            
            result_df['EMA_Medium_Long_Crossover'] = np.where(
                result_df[f'EMA_{self.ema_medium_period}'] > result_df[f'EMA_{self.ema_long_period}'], 1,
                np.where(result_df[f'EMA_{self.ema_medium_period}'] < result_df[f'EMA_{self.ema_long_period}'], -1, 0)
            )
            
            # Calculate EMA slopes
            result_df[f'EMA_{self.ema_short_period}_Slope'] = result_df[f'EMA_{self.ema_short_period}'].diff() / result_df[f'EMA_{self.ema_short_period}'].shift(1) * 100
            result_df[f'EMA_{self.ema_medium_period}_Slope'] = result_df[f'EMA_{self.ema_medium_period}'].diff() / result_df[f'EMA_{self.ema_medium_period}'].shift(1) * 100
            result_df[f'EMA_{self.ema_long_period}_Slope'] = result_df[f'EMA_{self.ema_long_period}'].diff() / result_df[f'EMA_{self.ema_long_period}'].shift(1) * 100
            
            # Calculate EMA-based market regime
            # Bullish: Short > Medium > Long and positive slopes
            # Bearish: Short < Medium < Long and negative slopes
            # Sideways: EMAs are close to each other and slopes are near zero
            
            # Calculate EMA distances
            result_df['EMA_Short_Medium_Distance'] = (result_df[f'EMA_{self.ema_short_period}'] - result_df[f'EMA_{self.ema_medium_period}']) / result_df[f'EMA_{self.ema_medium_period}'] * 100
            result_df['EMA_Medium_Long_Distance'] = (result_df[f'EMA_{self.ema_medium_period}'] - result_df[f'EMA_{self.ema_long_period}']) / result_df[f'EMA_{self.ema_long_period}'] * 100
            
            # Define EMA-based market regime
            conditions = [
                # Strong Bullish
                (result_df[f'EMA_{self.ema_short_period}'] > result_df[f'EMA_{self.ema_medium_period}']) &
                (result_df[f'EMA_{self.ema_medium_period}'] > result_df[f'EMA_{self.ema_long_period}']) &
                (result_df[f'EMA_{self.ema_short_period}_Slope'] > 0.1) &
                (result_df[f'EMA_{self.ema_medium_period}_Slope'] > 0.05),
                
                # Bullish
                (result_df[f'EMA_{self.ema_short_period}'] > result_df[f'EMA_{self.ema_medium_period}']) &
                (result_df[f'EMA_{self.ema_medium_period}'] > result_df[f'EMA_{self.ema_long_period}']),
                
                # Strong Bearish
                (result_df[f'EMA_{self.ema_short_period}'] < result_df[f'EMA_{self.ema_medium_period}']) &
                (result_df[f'EMA_{self.ema_medium_period}'] < result_df[f'EMA_{self.ema_long_period}']) &
                (result_df[f'EMA_{self.ema_short_period}_Slope'] < -0.1) &
                (result_df[f'EMA_{self.ema_medium_period}_Slope'] < -0.05),
                
                # Bearish
                (result_df[f'EMA_{self.ema_short_period}'] < result_df[f'EMA_{self.ema_medium_period}']) &
                (result_df[f'EMA_{self.ema_medium_period}'] < result_df[f'EMA_{self.ema_long_period}']),
                
                # Sideways
                (result_df['EMA_Short_Medium_Distance'].abs() < 0.5) &
                (result_df['EMA_Medium_Long_Distance'].abs() < 0.5) &
                (result_df[f'EMA_{self.ema_short_period}_Slope'].abs() < 0.05)
            ]
            
            choices = ['Strong_Bullish', 'Bullish', 'Strong_Bearish', 'Bearish', 'Sideways']
            result_df['EMA_Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            logger.info("Calculated EMA indicators")
            
        except Exception as e:
            logger.error(f"Error calculating EMA indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def calculate_vwap_indicators(self, df):
        """
        Calculate VWAP indicators
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with VWAP indicators
        """
        logger.info("Calculating VWAP indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have required price data
        required_columns = ['Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for VWAP calculation: {missing_columns}")
            return result_df
        
        try:
            # Calculate typical price
            if all(col in result_df.columns for col in ['High', 'Low']):
                result_df['Typical_Price'] = (result_df['High'] + result_df['Low'] + result_df['Close']) / 3
            else:
                result_df['Typical_Price'] = result_df['Close']
            
            # Calculate VWAP
            result_df['Volume_Price'] = result_df['Typical_Price'] * result_df['Volume']
            
            # Calculate rolling VWAP
            result_df['Cumulative_Volume_Price'] = result_df['Volume_Price'].rolling(window=self.vwap_period).sum()
            result_df['Cumulative_Volume'] = result_df['Volume'].rolling(window=self.vwap_period).sum()
            result_df['VWAP'] = result_df['Cumulative_Volume_Price'] / result_df['Cumulative_Volume']
            
            # Calculate VWAP distance
            result_df['VWAP_Distance'] = (result_df['Close'] - result_df['VWAP']) / result_df['VWAP'] * 100
            
            # Calculate VWAP slope
            result_df['VWAP_Slope'] = result_df['VWAP'].diff() / result_df['VWAP'].shift(1) * 100
            
            # Define VWAP-based market regime
            conditions = [
                # Strong Bullish
                (result_df['Close'] > result_df['VWAP']) &
                (result_df['VWAP_Distance'] > 1.0) &
                (result_df['VWAP_Slope'] > 0.1),
                
                # Bullish
                (result_df['Close'] > result_df['VWAP']) &
                (result_df['VWAP_Slope'] > 0),
                
                # Strong Bearish
                (result_df['Close'] < result_df['VWAP']) &
                (result_df['VWAP_Distance'] < -1.0) &
                (result_df['VWAP_Slope'] < -0.1),
                
                # Bearish
                (result_df['Close'] < result_df['VWAP']) &
                (result_df['VWAP_Slope'] < 0),
                
                # Sideways
                (result_df['VWAP_Distance'].abs() < 0.5) &
                (result_df['VWAP_Slope'].abs() < 0.05)
            ]
            
            choices = ['Strong_Bullish', 'Bullish', 'Strong_Bearish', 'Bearish', 'Sideways']
            result_df['VWAP_Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            # Clean up intermediate columns
            result_df.drop(['Volume_Price', 'Cumulative_Volume_Price', 'Cumulative_Volume'], axis=1, inplace=True)
            
            logger.info("Calculated VWAP indicators")
            
        except Exception as e:
            logger.error(f"Error calculating VWAP indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def calculate_rsi_indicator(self, df):
        """
        Calculate RSI indicator
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with RSI indicator
        """
        logger.info("Calculating RSI indicator")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have Close price
        if 'Close' not in result_df.columns:
            logger.error("No Close price data available for RSI calculation")
            return result_df
        
        try:
            # Calculate RSI
            result_df['RSI'] = talib.RSI(result_df['Close'], timeperiod=self.rsi_period)
            
            # Define RSI-based market regime
            conditions = [
                # Strong Bullish
                (result_df['RSI'] > 70) &
                (result_df['RSI'].shift(1) <= 70),
                
                # Bullish
                (result_df['RSI'] > 50) &
                (result_df['RSI'] < 70),
                
                # Strong Bearish
                (result_df['RSI'] < 30) &
                (result_df['RSI'].shift(1) >= 30),
                
                # Bearish
                (result_df['RSI'] < 50) &
                (result_df['RSI'] > 30),
                
                # Overbought
                (result_df['RSI'] > 70),
                
                # Oversold
                (result_df['RSI'] < 30)
            ]
            
            choices = ['Strong_Bullish', 'Bullish', 'Strong_Bearish', 'Bearish', 'Overbought', 'Oversold']
            result_df['RSI_Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            logger.info("Calculated RSI indicator")
            
        except Exception as e:
            logger.error(f"Error calculating RSI indicator: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def calculate_macd_indicator(self, df):
        """
        Calculate MACD indicator
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with MACD indicator
        """
        logger.info("Calculating MACD indicator")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have Close price
        if 'Close' not in result_df.columns:
            logger.error("No Close price data available for MACD calculation")
            return result_df
        
        try:
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(
                result_df['Close'],
                fastperiod=self.macd_fast_period,
                slowperiod=self.macd_slow_period,
                signalperiod=self.macd_signal_period
            )
            
            result_df['MACD'] = macd
            result_df['MACD_Signal'] = macd_signal
            result_df['MACD_Hist'] = macd_hist
            
            # Calculate MACD crossovers
            result_df['MACD_Crossover'] = np.where(
                (result_df['MACD'] > result_df['MACD_Signal']) & (result_df['MACD'].shift(1) <= result_df['MACD_Signal'].shift(1)), 1,
                np.where((result_df['MACD'] < result_df['MACD_Signal']) & (result_df['MACD'].shift(1) >= result_df['MACD_Signal'].shift(1)), -1, 0)
            )
            
            # Define MACD-based market regime
            conditions = [
                # Strong Bullish
                (result_df['MACD'] > result_df['MACD_Signal']) &
                (result_df['MACD'] > 0) &
                (result_df['MACD_Hist'] > 0) &
                (result_df['MACD_Hist'] > result_df['MACD_Hist'].shift(1)),
                
                # Bullish
                (result_df['MACD'] > result_df['MACD_Signal']) &
                (result_df['MACD_Hist'] > 0),
                
                # Strong Bearish
                (result_df['MACD'] < result_df['MACD_Signal']) &
                (result_df['MACD'] < 0) &
                (result_df['MACD_Hist'] < 0) &
                (result_df['MACD_Hist'] < result_df['MACD_Hist'].shift(1)),
                
                # Bearish
                (result_df['MACD'] < result_df['MACD_Signal']) &
                (result_df['MACD_Hist'] < 0),
                
                # Bullish Crossover
                (result_df['MACD_Crossover'] == 1),
                
                # Bearish Crossover
                (result_df['MACD_Crossover'] == -1)
            ]
            
            choices = ['Strong_Bullish', 'Bullish', 'Strong_Bearish', 'Bearish', 'Bullish_Crossover', 'Bearish_Crossover']
            result_df['MACD_Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            logger.info("Calculated MACD indicator")
            
        except Exception as e:
            logger.error(f"Error calculating MACD indicator: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def calculate_bollinger_bands(self, df):
        """
        Calculate Bollinger Bands
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with Bollinger Bands
        """
        logger.info("Calculating Bollinger Bands")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have Close price
        if 'Close' not in result_df.columns:
            logger.error("No Close price data available for Bollinger Bands calculation")
            return result_df
        
        try:
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                result_df['Close'],
                timeperiod=self.bollinger_period,
                nbdevup=self.bollinger_std_dev,
                nbdevdn=self.bollinger_std_dev,
                matype=0
            )
            
            result_df['BB_Upper'] = upper
            result_df['BB_Middle'] = middle
            result_df['BB_Lower'] = lower
            
            # Calculate Bollinger Band width
            result_df['BB_Width'] = (result_df['BB_Upper'] - result_df['BB_Lower']) / result_df['BB_Middle'] * 100
            
            # Calculate Bollinger Band position
            result_df['BB_Position'] = (result_df['Close'] - result_df['BB_Lower']) / (result_df['BB_Upper'] - result_df['BB_Lower'])
            
            # Define Bollinger Bands-based market regime
            conditions = [
                # Strong Bullish
                (result_df['Close'] > result_df['BB_Upper']) &
                (result_df['Close'].shift(1) <= result_df['BB_Upper'].shift(1)),
                
                # Bullish
                (result_df['Close'] > result_df['BB_Middle']) &
                (result_df['Close'] < result_df['BB_Upper']),
                
                # Strong Bearish
                (result_df['Close'] < result_df['BB_Lower']) &
                (result_df['Close'].shift(1) >= result_df['BB_Lower'].shift(1)),
                
                # Bearish
                (result_df['Close'] < result_df['BB_Middle']) &
                (result_df['Close'] > result_df['BB_Lower']),
                
                # Squeeze (low volatility)
                (result_df['BB_Width'] < result_df['BB_Width'].rolling(window=20).mean() * 0.8),
                
                # Expansion (high volatility)
                (result_df['BB_Width'] > result_df['BB_Width'].rolling(window=20).mean() * 1.2)
            ]
            
            choices = ['Strong_Bullish', 'Bullish', 'Strong_Bearish', 'Bearish', 'Squeeze', 'Expansion']
            result_df['BB_Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            logger.info("Calculated Bollinger Bands")
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def calculate_atr_indicator(self, df):
        """
        Calculate ATR indicator
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with ATR indicator
        """
        logger.info("Calculating ATR indicator")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have required price data
        required_columns = ['High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for ATR calculation: {missing_columns}")
            return result_df
        
        try:
            # Calculate ATR
            result_df['ATR'] = talib.ATR(
                result_df['High'],
                result_df['Low'],
                result_df['Close'],
                timeperiod=self.atr_period
            )
            
            # Calculate ATR percentage
            result_df['ATR_Pct'] = result_df['ATR'] / result_df['Close'] * 100
            
            # Calculate normalized ATR
            result_df['ATR_Normalized'] = result_df['ATR_Pct'] / result_df['ATR_Pct'].rolling(window=20).mean()
            
            # Define ATR-based market regime
            conditions = [
                # High Volatility
                (result_df['ATR_Normalized'] > 1.5),
                
                # Above Average Volatility
                (result_df['ATR_Normalized'] > 1.2) &
                (result_df['ATR_Normalized'] <= 1.5),
                
                # Below Average Volatility
                (result_df['ATR_Normalized'] < 0.8) &
                (result_df['ATR_Normalized'] >= 0.5),
                
                # Low Volatility
                (result_df['ATR_Normalized'] < 0.5)
            ]
            
            choices = ['High_Volatility', 'Above_Average_Volatility', 'Below_Average_Volatility', 'Low_Volatility']
            result_df['ATR_Market_Regime'] = np.select(conditions, choices, default='Average_Volatility')
            
            logger.info("Calculated ATR indicator")
            
        except Exception as e:
            logger.error(f"Error calculating ATR indicator: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def calculate_combined_technical_regime(self, df):
        """
        Calculate combined technical market regime
        
        Args:
            df (pd.DataFrame): Input dataframe with technical indicators
            
        Returns:
            pd.DataFrame: Dataframe with combined technical market regime
        """
        logger.info("Calculating combined technical market regime")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we have required regime columns
        regime_columns = ['EMA_Market_Regime', 'VWAP_Market_Regime', 'RSI_Market_Regime', 'MACD_Market_Regime', 'BB_Market_Regime']
        available_regimes = [col for col in regime_columns if col in result_df.columns]
        
        if not available_regimes:
            logger.error("No market regime columns available for combined regime calculation")
            return result_df
        
        try:
            # Create regime score mapping
            regime_scores = {
                'Strong_Bullish': 2,
                'Bullish': 1,
                'Bullish_Crossover': 1,
                'Neutral': 0,
                'Sideways': 0,
                'Bearish': -1,
                'Bearish_Crossover': -1,
                'Strong_Bearish': -2,
                'Overbought': 0,
                'Oversold': 0,
                'Squeeze': 0,
                'Expansion': 0,
                'High_Volatility': 0,
                'Above_Average_Volatility': 0,
                'Average_Volatility': 0,
                'Below_Average_Volatility': 0,
                'Low_Volatility': 0
            }
            
            # Calculate regime scores for each indicator
            for regime_col in available_regimes:
                score_col = f"{regime_col}_Score"
                result_df[score_col] = result_df[regime_col].map(regime_scores)
            
            # Calculate average regime score
            score_columns = [f"{regime_col}_Score" for regime_col in available_regimes]
            result_df['Technical_Regime_Score'] = result_df[score_columns].mean(axis=1)
            
            # Define combined technical market regime
            conditions = [
                # Strong Bullish
                (result_df['Technical_Regime_Score'] >= 1.5),
                
                # Bullish
                (result_df['Technical_Regime_Score'] >= 0.5) &
                (result_df['Technical_Regime_Score'] < 1.5),
                
                # Strong Bearish
                (result_df['Technical_Regime_Score'] <= -1.5),
                
                # Bearish
                (result_df['Technical_Regime_Score'] <= -0.5) &
                (result_df['Technical_Regime_Score'] > -1.5),
                
                # Neutral
                (result_df['Technical_Regime_Score'] > -0.5) &
                (result_df['Technical_Regime_Score'] < 0.5)
            ]
            
            choices = ['Strong_Bullish', 'Bullish', 'Strong_Bearish', 'Bearish', 'Neutral']
            result_df['Technical_Market_Regime'] = np.select(conditions, choices, default='Neutral')
            
            # Calculate regime confidence
            # Higher confidence when more indicators agree
            regime_count = len(available_regimes)
            
            # Count how many indicators agree with the combined regime
            result_df['Technical_Regime_Agreement'] = 0
            
            for regime_col in available_regimes:
                # Check if the indicator regime matches the combined regime direction
                bullish_match = ((result_df[regime_col].isin(['Strong_Bullish', 'Bullish', 'Bullish_Crossover'])) & 
                                (result_df['Technical_Market_Regime'].isin(['Strong_Bullish', 'Bullish'])))
                
                bearish_match = ((result_df[regime_col].isin(['Strong_Bearish', 'Bearish', 'Bearish_Crossover'])) & 
                                (result_df['Technical_Market_Regime'].isin(['Strong_Bearish', 'Bearish'])))
                
                neutral_match = ((result_df[regime_col].isin(['Neutral', 'Sideways', 'Squeeze', 'Average_Volatility'])) & 
                                (result_df['Technical_Market_Regime'] == 'Neutral'))
                
                result_df['Technical_Regime_Agreement'] += bullish_match | bearish_match | neutral_match
            
            # Calculate confidence as percentage of agreeing indicators
            result_df['Technical_Regime_Confidence'] = result_df['Technical_Regime_Agreement'] / regime_count
            
            logger.info("Calculated combined technical market regime")
            
        except Exception as e:
            logger.error(f"Error calculating combined technical market regime: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def visualize_technical_indicators(self, df):
        """
        Visualize technical indicators
        
        Args:
            df (pd.DataFrame): Dataframe with technical indicators
        """
        logger.info("Visualizing technical indicators")
        
        # Check if we have required columns
        if 'Close' not in df.columns:
            logger.error("No Close price data available for visualization")
            return
        
        try:
            # Create price chart with EMAs
            if all(col in df.columns for col in [f'EMA_{self.ema_short_period}', f'EMA_{self.ema_medium_period}', f'EMA_{self.ema_long_period}']):
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['Close'], label='Close', color='black', alpha=0.5)
                plt.plot(df[f'EMA_{self.ema_short_period}'], label=f'EMA {self.ema_short_period}', linewidth=1.5)
                plt.plot(df[f'EMA_{self.ema_medium_period}'], label=f'EMA {self.ema_medium_period}', linewidth=1.5)
                plt.plot(df[f'EMA_{self.ema_long_period}'], label=f'EMA {self.ema_long_period}', linewidth=1.5)
                
                plt.title('Price with EMAs')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                ema_plot_path = os.path.join(self.output_dir, 'price_with_emas.png')
                plt.savefig(ema_plot_path)
                logger.info(f"Saved price with EMAs plot to {ema_plot_path}")
                
                plt.close()
            
            # Create VWAP chart
            if 'VWAP' in df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['Close'], label='Close', color='black', alpha=0.5)
                plt.plot(df['VWAP'], label='VWAP', linewidth=1.5, color='purple')
                
                plt.title('Price with VWAP')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                vwap_plot_path = os.path.join(self.output_dir, 'price_with_vwap.png')
                plt.savefig(vwap_plot_path)
                logger.info(f"Saved price with VWAP plot to {vwap_plot_path}")
                
                plt.close()
            
            # Create RSI chart
            if 'RSI' in df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['RSI'], label='RSI', linewidth=1.5, color='blue')
                plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                plt.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
                
                plt.title('RSI Indicator')
                plt.xlabel('Time')
                plt.ylabel('RSI')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                rsi_plot_path = os.path.join(self.output_dir, 'rsi_indicator.png')
                plt.savefig(rsi_plot_path)
                logger.info(f"Saved RSI indicator plot to {rsi_plot_path}")
                
                plt.close()
            
            # Create MACD chart
            if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                plt.figure(figsize=(12, 6))
                
                plt.subplot(2, 1, 1)
                plt.plot(df['Close'], label='Close', color='black', alpha=0.5)
                plt.title('Price')
                plt.grid(alpha=0.3)
                
                plt.subplot(2, 1, 2)
                plt.plot(df['MACD'], label='MACD', linewidth=1.5, color='blue')
                plt.plot(df['MACD_Signal'], label='Signal', linewidth=1.5, color='red')
                
                # Plot histogram
                positive_hist = df['MACD_Hist'].copy()
                negative_hist = df['MACD_Hist'].copy()
                positive_hist[positive_hist <= 0] = np.nan
                negative_hist[negative_hist > 0] = np.nan
                
                plt.bar(range(len(df)), positive_hist, color='green', alpha=0.5, label='Positive')
                plt.bar(range(len(df)), negative_hist, color='red', alpha=0.5, label='Negative')
                
                plt.title('MACD Indicator')
                plt.xlabel('Time')
                plt.ylabel('MACD')
                plt.legend()
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                macd_plot_path = os.path.join(self.output_dir, 'macd_indicator.png')
                plt.savefig(macd_plot_path)
                logger.info(f"Saved MACD indicator plot to {macd_plot_path}")
                
                plt.close()
            
            # Create Bollinger Bands chart
            if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['Close'], label='Close', color='black', alpha=0.5)
                plt.plot(df['BB_Upper'], label='Upper Band', linewidth=1.5, color='red')
                plt.plot(df['BB_Middle'], label='Middle Band', linewidth=1.5, color='blue')
                plt.plot(df['BB_Lower'], label='Lower Band', linewidth=1.5, color='green')
                
                plt.title('Bollinger Bands')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                bb_plot_path = os.path.join(self.output_dir, 'bollinger_bands.png')
                plt.savefig(bb_plot_path)
                logger.info(f"Saved Bollinger Bands plot to {bb_plot_path}")
                
                plt.close()
                
                # Create Bollinger Band Width chart
                if 'BB_Width' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    plt.plot(df['BB_Width'], label='BB Width', linewidth=1.5, color='purple')
                    
                    plt.title('Bollinger Band Width')
                    plt.xlabel('Time')
                    plt.ylabel('Width (%)')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    bb_width_plot_path = os.path.join(self.output_dir, 'bollinger_band_width.png')
                    plt.savefig(bb_width_plot_path)
                    logger.info(f"Saved Bollinger Band Width plot to {bb_width_plot_path}")
                    
                    plt.close()
            
            # Create ATR chart
            if 'ATR' in df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['ATR'], label='ATR', linewidth=1.5, color='orange')
                
                plt.title('Average True Range (ATR)')
                plt.xlabel('Time')
                plt.ylabel('ATR')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                atr_plot_path = os.path.join(self.output_dir, 'atr_indicator.png')
                plt.savefig(atr_plot_path)
                logger.info(f"Saved ATR indicator plot to {atr_plot_path}")
                
                plt.close()
                
                # Create ATR Percentage chart
                if 'ATR_Pct' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    plt.plot(df['ATR_Pct'], label='ATR %', linewidth=1.5, color='orange')
                    
                    plt.title('ATR Percentage')
                    plt.xlabel('Time')
                    plt.ylabel('ATR %')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    atr_pct_plot_path = os.path.join(self.output_dir, 'atr_percentage.png')
                    plt.savefig(atr_pct_plot_path)
                    logger.info(f"Saved ATR percentage plot to {atr_pct_plot_path}")
                    
                    plt.close()
            
            # Create Technical Market Regime chart
            if 'Technical_Market_Regime' in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Count occurrences of each regime
                regime_counts = df['Technical_Market_Regime'].value_counts()
                
                # Create a colormap for regimes
                regime_colors = {
                    'Strong_Bullish': 'green',
                    'Bullish': 'lightgreen',
                    'Neutral': 'gray',
                    'Bearish': 'lightcoral',
                    'Strong_Bearish': 'red'
                }
                
                # Plot regime distribution
                bars = plt.bar(regime_counts.index, regime_counts.values, 
                              color=[regime_colors.get(regime, 'blue') for regime in regime_counts.index])
                
                plt.title('Technical Market Regime Distribution')
                plt.xlabel('Regime')
                plt.ylabel('Count')
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save plot
                regime_plot_path = os.path.join(self.output_dir, 'technical_market_regime_distribution.png')
                plt.savefig(regime_plot_path)
                logger.info(f"Saved technical market regime distribution plot to {regime_plot_path}")
                
                plt.close()
                
                # Create regime over time chart
                plt.figure(figsize=(12, 6))
                
                # Plot price
                plt.plot(df['Close'], label='Close', alpha=0.5, color='black')
                
                # Plot regime as background color
                x_points = range(len(df))
                
                # Plot colored background for each regime
                for regime, color in regime_colors.items():
                    mask = df['Technical_Market_Regime'] == regime
                    if mask.any():
                        plt.fill_between(x_points, 0, df['Close'].max(), where=mask, color=color, alpha=0.2, label=regime)
                
                plt.title('Technical Market Regime Over Time')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                regime_time_plot_path = os.path.join(self.output_dir, 'technical_market_regime_time.png')
                plt.savefig(regime_time_plot_path)
                logger.info(f"Saved technical market regime over time plot to {regime_time_plot_path}")
                
                plt.close()
                
                # Create regime confidence chart
                if 'Technical_Regime_Confidence' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    plt.plot(df['Technical_Regime_Confidence'], label='Confidence', linewidth=1.5, color='blue')
                    
                    plt.title('Technical Market Regime Confidence')
                    plt.xlabel('Time')
                    plt.ylabel('Confidence')
                    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50% Confidence')
                    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='80% Confidence')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    confidence_plot_path = os.path.join(self.output_dir, 'technical_regime_confidence.png')
                    plt.savefig(confidence_plot_path)
                    logger.info(f"Saved technical regime confidence plot to {confidence_plot_path}")
                    
                    plt.close()
            
            logger.info("Completed visualization of technical indicators")
            
        except Exception as e:
            logger.error(f"Error visualizing technical indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_technical_indicators(self):
        """
        Test technical indicators
        
        Returns:
            pd.DataFrame: Dataframe with technical indicators
        """
        logger.info("Testing technical indicators")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None
        
        # Prepare price data
        price_df = self.prepare_price_data(df)
        
        if price_df is None:
            logger.error("Failed to prepare price data")
            return None
        
        # Calculate technical indicators
        result_df = price_df.copy()
        
        # Calculate EMA indicators
        result_df = self.calculate_ema_indicators(result_df)
        
        # Calculate VWAP indicators
        result_df = self.calculate_vwap_indicators(result_df)
        
        # Calculate RSI indicator
        result_df = self.calculate_rsi_indicator(result_df)
        
        # Calculate MACD indicator
        result_df = self.calculate_macd_indicator(result_df)
        
        # Calculate Bollinger Bands
        result_df = self.calculate_bollinger_bands(result_df)
        
        # Calculate ATR indicator
        result_df = self.calculate_atr_indicator(result_df)
        
        # Calculate combined technical market regime
        result_df = self.calculate_combined_technical_regime(result_df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "technical_indicators_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved technical indicators results to {output_path}")
        
        # Visualize results
        self.visualize_technical_indicators(result_df)
        
        # Log summary statistics
        if 'Technical_Market_Regime' in result_df.columns:
            regime_counts = result_df['Technical_Market_Regime'].value_counts()
            logger.info(f"Technical Market Regime distribution: {regime_counts.to_dict()}")
        
        logger.info("Technical indicators testing completed")
        
        return result_df

def main():
    """
    Main function to run the technical indicators testing
    """
    logger.info("Starting technical indicators testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/technical_indicators',
        'ema_short_period': 9,
        'ema_medium_period': 21,
        'ema_long_period': 50,
        'vwap_period': 14,
        'rsi_period': 14,
        'macd_fast_period': 12,
        'macd_slow_period': 26,
        'macd_signal_period': 9,
        'bollinger_period': 20,
        'bollinger_std_dev': 2,
        'atr_period': 14
    }
    
    # Create technical indicators tester
    tester = TechnicalIndicatorsTester(config)
    
    # Test technical indicators
    result_df = tester.test_technical_indicators()
    
    logger.info("Technical indicators testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
