import pandas as pd
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)

class EMAIndicators:
    """
    EMA Indicators Calculator.
    
    This class calculates Exponential Moving Average (EMA) indicators
    across multiple timeframes and provides signals based on EMA relationships.
    """
    
    def __init__(self, config=None):
        """
        Initialize EMA Indicators Calculator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # EMA periods
        self.short_period = int(self.config.get('short_period', 20))
        self.mid_period = int(self.config.get('mid_period', 50))
        self.long_period = int(self.config.get('long_period', 200))
        
        # Timeframes in minutes
        self.timeframes = self.config.get('timeframes', [15, 10, 5, 3])
        
        logger.info(f"Initialized EMA Indicators with periods: {self.short_period}, {self.mid_period}, {self.long_period}")
    
    def calculate_ema(self, df, price_column='Close', period=20):
        """
        Calculate EMA for a given period.
        
        Args:
            df (pd.DataFrame): Price data
            price_column (str): Column name for price data
            period (int): EMA period
            
        Returns:
            pd.Series: EMA values
        """
        return df[price_column].ewm(span=period, adjust=False).mean()
    
    def calculate_all_emas(self, df, price_column='Close'):
        """
        Calculate all EMAs (short, mid, long).
        
        Args:
            df (pd.DataFrame): Price data
            price_column (str): Column name for price data
            
        Returns:
            pd.DataFrame: DataFrame with EMA columns added
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate EMAs
        result_df[f'EMA_{self.short_period}'] = self.calculate_ema(df, price_column, self.short_period)
        result_df[f'EMA_{self.mid_period}'] = self.calculate_ema(df, price_column, self.mid_period)
        result_df[f'EMA_{self.long_period}'] = self.calculate_ema(df, price_column, self.long_period)
        
        return result_df
    
    def get_ema_signal(self, df, price_column='Close'):
        """
        Get EMA signal based on price and EMA relationships.
        
        Args:
            df (pd.DataFrame): Price data with EMAs
            price_column (str): Column name for price data
            
        Returns:
            pd.Series: EMA signal (-1 to 1)
        """
        # Calculate EMAs if they don't exist
        if f'EMA_{self.short_period}' not in df.columns:
            df = self.calculate_all_emas(df, price_column)
        
        # Initialize signal series
        signal = pd.Series(0, index=df.index)
        
        # Price above all EMAs (strongest bullish)
        mask = (df[price_column] > df[f'EMA_{self.short_period}']) & \
               (df[price_column] > df[f'EMA_{self.mid_period}']) & \
               (df[price_column] > df[f'EMA_{self.long_period}'])
        signal[mask] = 1.0
        
        # Price above mid and long EMAs
        mask = (df[price_column] <= df[f'EMA_{self.short_period}']) & \
               (df[price_column] > df[f'EMA_{self.mid_period}']) & \
               (df[price_column] > df[f'EMA_{self.long_period}'])
        signal[mask] = 0.75
        
        # Price above long EMA only
        mask = (df[price_column] <= df[f'EMA_{self.short_period}']) & \
               (df[price_column] <= df[f'EMA_{self.mid_period}']) & \
               (df[price_column] > df[f'EMA_{self.long_period}'])
        signal[mask] = 0.25
        
        # Price below all EMAs (strongest bearish)
        mask = (df[price_column] < df[f'EMA_{self.short_period}']) & \
               (df[price_column] < df[f'EMA_{self.mid_period}']) & \
               (df[price_column] < df[f'EMA_{self.long_period}'])
        signal[mask] = -1.0
        
        # Price below mid and long EMAs
        mask = (df[price_column] >= df[f'EMA_{self.short_period}']) & \
               (df[price_column] < df[f'EMA_{self.mid_period}']) & \
               (df[price_column] < df[f'EMA_{self.long_period}'])
        signal[mask] = -0.75
        
        # Price below long EMA only
        mask = (df[price_column] >= df[f'EMA_{self.short_period}']) & \
               (df[price_column] >= df[f'EMA_{self.mid_period}']) & \
               (df[price_column] < df[f'EMA_{self.long_period}'])
        signal[mask] = -0.25
        
        return signal
    
    def get_ema_alignment_signal(self, df):
        """
        Get EMA alignment signal based on EMA relationships.
        
        Args:
            df (pd.DataFrame): Price data with EMAs
            
        Returns:
            pd.Series: EMA alignment signal (-1 to 1)
        """
        # Calculate EMAs if they don't exist
        if f'EMA_{self.short_period}' not in df.columns:
            df = self.calculate_all_emas(df)
        
        # Initialize signal series
        signal = pd.Series(0, index=df.index)
        
        # Bullish alignment (short > mid > long)
        mask = (df[f'EMA_{self.short_period}'] > df[f'EMA_{self.mid_period}']) & \
               (df[f'EMA_{self.mid_period}'] > df[f'EMA_{self.long_period}'])
        signal[mask] = 1.0
        
        # Partial bullish alignment (short > mid, mid <= long)
        mask = (df[f'EMA_{self.short_period}'] > df[f'EMA_{self.mid_period}']) & \
               (df[f'EMA_{self.mid_period}'] <= df[f'EMA_{self.long_period}'])
        signal[mask] = 0.5
        
        # Bearish alignment (short < mid < long)
        mask = (df[f'EMA_{self.short_period}'] < df[f'EMA_{self.mid_period}']) & \
               (df[f'EMA_{self.mid_period}'] < df[f'EMA_{self.long_period}'])
        signal[mask] = -1.0
        
        # Partial bearish alignment (short < mid, mid >= long)
        mask = (df[f'EMA_{self.short_period}'] < df[f'EMA_{self.mid_period}']) & \
               (df[f'EMA_{self.mid_period}'] >= df[f'EMA_{self.long_period}'])
        signal[mask] = -0.5
        
        return signal
    
    def get_ema_crossover_signal(self, df):
        """
        Get EMA crossover signal.
        
        Args:
            df (pd.DataFrame): Price data with EMAs
            
        Returns:
            pd.Series: EMA crossover signal (-1 to 1)
        """
        # Calculate EMAs if they don't exist
        if f'EMA_{self.short_period}' not in df.columns:
            df = self.calculate_all_emas(df)
        
        # Initialize signal series
        signal = pd.Series(0, index=df.index)
        
        # Calculate previous values
        prev_short = df[f'EMA_{self.short_period}'].shift(1)
        prev_mid = df[f'EMA_{self.mid_period}'].shift(1)
        
        # Bullish crossover (short crosses above mid)
        mask = (df[f'EMA_{self.short_period}'] > df[f'EMA_{self.mid_period}']) & \
               (prev_short <= prev_mid)
        signal[mask] = 1.0
        
        # Bearish crossover (short crosses below mid)
        mask = (df[f'EMA_{self.short_period}'] < df[f'EMA_{self.mid_period}']) & \
               (prev_short >= prev_mid)
        signal[mask] = -1.0
        
        return signal
    
    def get_multi_timeframe_signal(self, data_dict, price_column='Close'):
        """
        Get multi-timeframe EMA signal.
        
        Args:
            data_dict (dict): Dictionary of DataFrames for different timeframes
            price_column (str): Column name for price data
            
        Returns:
            float: Combined EMA signal (-1 to 1)
        """
        signals = []
        weights = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            if timeframe in data_dict:
                df = data_dict[timeframe]
                df = self.calculate_all_emas(df, price_column)
                
                # Get signals
                price_signal = self.get_ema_signal(df, price_column)
                alignment_signal = self.get_ema_alignment_signal(df)
                crossover_signal = self.get_ema_crossover_signal(df)
                
                # Combine signals for this timeframe
                combined_signal = 0.5 * price_signal + 0.3 * alignment_signal + 0.2 * crossover_signal
                
                # Get the latest signal
                latest_signal = combined_signal.iloc[-1]
                
                # Add to signals list with weight based on timeframe
                # Higher weight for shorter timeframes
                weight = 1.0 / timeframe
                signals.append(latest_signal)
                weights.append(weight)
        
        # If no signals, return 0
        if not signals:
            return 0.0
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
        
        # Clip to [-1, 1]
        return max(-1.0, min(1.0, weighted_signal))
