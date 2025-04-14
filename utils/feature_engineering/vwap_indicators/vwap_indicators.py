import pandas as pd
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)

class VWAPIndicators:
    """
    VWAP Indicators Calculator.
    
    This class calculates Volume Weighted Average Price (VWAP) indicators
    across multiple timeframes and provides signals based on VWAP relationships.
    """
    
    def __init__(self, config=None):
        """
        Initialize VWAP Indicators Calculator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Timeframes in minutes
        self.timeframes = self.config.get('timeframes', [15, 10, 5, 3])
        
        # VWAP bands
        self.band_multipliers = self.config.get('band_multipliers', [1.0, 1.5, 2.0])
        
        logger.info(f"Initialized VWAP Indicators with timeframes: {self.timeframes}")
    
    def calculate_vwap(self, df, high_col='High', low_col='Low', close_col='Close', volume_col='Volume'):
        """
        Calculate VWAP for a given DataFrame.
        
        Args:
            df (pd.DataFrame): Price and volume data
            high_col (str): Column name for high price
            low_col (str): Column name for low price
            close_col (str): Column name for close price
            volume_col (str): Column name for volume
            
        Returns:
            pd.Series: VWAP values
        """
        # Calculate typical price
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        
        # Calculate VWAP
        vwap = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()
        
        return vwap
    
    def calculate_vwap_bands(self, df, vwap_col='VWAP', high_col='High', low_col='Low', close_col='Close', volume_col='Volume'):
        """
        Calculate VWAP bands.
        
        Args:
            df (pd.DataFrame): Price and volume data with VWAP
            vwap_col (str): Column name for VWAP
            high_col (str): Column name for high price
            low_col (str): Column name for low price
            close_col (str): Column name for close price
            volume_col (str): Column name for volume
            
        Returns:
            pd.DataFrame: DataFrame with VWAP bands added
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate VWAP if it doesn't exist
        if vwap_col not in result_df.columns:
            result_df[vwap_col] = self.calculate_vwap(df, high_col, low_col, close_col, volume_col)
        
        # Calculate standard deviation of price from VWAP
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        std_dev = np.sqrt(((typical_price - result_df[vwap_col]) ** 2 * df[volume_col]).cumsum() / df[volume_col].cumsum())
        
        # Calculate VWAP bands
        for multiplier in self.band_multipliers:
            result_df[f'{vwap_col}_upper_{multiplier}'] = result_df[vwap_col] + multiplier * std_dev
            result_df[f'{vwap_col}_lower_{multiplier}'] = result_df[vwap_col] - multiplier * std_dev
        
        return result_df
    
    def get_vwap_signal(self, df, price_col='Close', vwap_col='VWAP'):
        """
        Get VWAP signal based on price and VWAP relationship.
        
        Args:
            df (pd.DataFrame): Price data with VWAP
            price_col (str): Column name for price data
            vwap_col (str): Column name for VWAP
            
        Returns:
            pd.Series: VWAP signal (-1 to 1)
        """
        # Initialize signal series
        signal = pd.Series(0, index=df.index)
        
        # Price above VWAP (bullish)
        mask = df[price_col] > df[vwap_col]
        signal[mask] = 1.0
        
        # Price below VWAP (bearish)
        mask = df[price_col] < df[vwap_col]
        signal[mask] = -1.0
        
        # Calculate distance from VWAP as percentage
        distance = (df[price_col] - df[vwap_col]) / df[vwap_col]
        
        # Scale signal by distance (closer to VWAP = weaker signal)
        signal = signal * distance.abs() * 5  # Scale factor of 5 to make signal more pronounced
        
        # Clip to [-1, 1]
        signal = signal.clip(-1, 1)
        
        return signal
    
    def get_vwap_band_signal(self, df, price_col='Close', vwap_col='VWAP'):
        """
        Get VWAP band signal based on price and VWAP bands relationship.
        
        Args:
            df (pd.DataFrame): Price data with VWAP bands
            price_col (str): Column name for price data
            vwap_col (str): Column name for VWAP
            
        Returns:
            pd.Series: VWAP band signal (-1 to 1)
        """
        # Initialize signal series
        signal = pd.Series(0, index=df.index)
        
        # Check if VWAP bands exist
        band_columns = [col for col in df.columns if col.startswith(f'{vwap_col}_upper_') or col.startswith(f'{vwap_col}_lower_')]
        
        if not band_columns:
            # If bands don't exist, calculate them
            df = self.calculate_vwap_bands(df, vwap_col)
        
        # Get sorted multipliers
        upper_bands = sorted([col for col in df.columns if col.startswith(f'{vwap_col}_upper_')], 
                            key=lambda x: float(x.split('_')[-1]))
        lower_bands = sorted([col for col in df.columns if col.startswith(f'{vwap_col}_lower_')], 
                            key=lambda x: float(x.split('_')[-1]), reverse=True)
        
        # Price above highest upper band (strongest bullish)
        if upper_bands:
            mask = df[price_col] > df[upper_bands[-1]]
            signal[mask] = 1.0
        
        # Price below lowest lower band (strongest bearish)
        if lower_bands:
            mask = df[price_col] < df[lower_bands[-1]]
            signal[mask] = -1.0
        
        # Price between bands
        for i, (upper, lower) in enumerate(zip(upper_bands, lower_bands)):
            # Calculate signal strength based on band position
            strength = 1.0 - (i / len(upper_bands))
            
            # Price between this upper band and next lower upper band
            if i < len(upper_bands) - 1:
                mask = (df[price_col] <= df[upper]) & (df[price_col] > df[upper_bands[i+1]])
                signal[mask] = strength
            
            # Price between this lower band and next higher lower band
            if i < len(lower_bands) - 1:
                mask = (df[price_col] >= df[lower]) & (df[price_col] < df[lower_bands[i+1]])
                signal[mask] = -strength
        
        return signal
    
    def get_vwap_crossover_signal(self, df, price_col='Close', vwap_col='VWAP'):
        """
        Get VWAP crossover signal.
        
        Args:
            df (pd.DataFrame): Price data with VWAP
            price_col (str): Column name for price data
            vwap_col (str): Column name for VWAP
            
        Returns:
            pd.Series: VWAP crossover signal (-1 to 1)
        """
        # Initialize signal series
        signal = pd.Series(0, index=df.index)
        
        # Calculate previous values
        prev_price = df[price_col].shift(1)
        prev_vwap = df[vwap_col].shift(1)
        
        # Bullish crossover (price crosses above VWAP)
        mask = (df[price_col] > df[vwap_col]) & (prev_price <= prev_vwap)
        signal[mask] = 1.0
        
        # Bearish crossover (price crosses below VWAP)
        mask = (df[price_col] < df[vwap_col]) & (prev_price >= prev_vwap)
        signal[mask] = -1.0
        
        return signal
    
    def get_multi_timeframe_signal(self, data_dict, high_col='High', low_col='Low', close_col='Close', volume_col='Volume'):
        """
        Get multi-timeframe VWAP signal.
        
        Args:
            data_dict (dict): Dictionary of DataFrames for different timeframes
            high_col (str): Column name for high price
            low_col (str): Column name for low price
            close_col (str): Column name for close price
            volume_col (str): Column name for volume
            
        Returns:
            float: Combined VWAP signal (-1 to 1)
        """
        signals = []
        weights = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            if timeframe in data_dict:
                df = data_dict[timeframe].copy()
                
                # Calculate VWAP
                df['VWAP'] = self.calculate_vwap(df, high_col, low_col, close_col, volume_col)
                
                # Calculate VWAP bands
                df = self.calculate_vwap_bands(df, 'VWAP', high_col, low_col, close_col, volume_col)
                
                # Get signals
                vwap_signal = self.get_vwap_signal(df, close_col, 'VWAP')
                band_signal = self.get_vwap_band_signal(df, close_col, 'VWAP')
                crossover_signal = self.get_vwap_crossover_signal(df, close_col, 'VWAP')
                
                # Combine signals for this timeframe
                combined_signal = 0.4 * vwap_signal + 0.4 * band_signal + 0.2 * crossover_signal
                
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
    
    def calculate_previous_day_vwap(self, df, date_col='Date', high_col='High', low_col='Low', close_col='Close', volume_col='Volume'):
        """
        Calculate previous day's VWAP for each day.
        
        Args:
            df (pd.DataFrame): Price and volume data
            date_col (str): Column name for date
            high_col (str): Column name for high price
            low_col (str): Column name for low price
            close_col (str): Column name for close price
            volume_col (str): Column name for volume
            
        Returns:
            pd.DataFrame: DataFrame with previous day's VWAP added
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure date column is datetime
        if date_col in result_df.columns:
            result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Group by date and calculate VWAP for each day
            daily_vwap = result_df.groupby(result_df[date_col].dt.date).apply(
                lambda x: self.calculate_vwap(x, high_col, low_col, close_col, volume_col).iloc[-1]
            )
            
            # Create a date-to-VWAP mapping
            vwap_map = dict(zip(daily_vwap.index, daily_vwap.values))
            
            # Add previous day's VWAP
            result_df['Prev_Day_VWAP'] = result_df[date_col].dt.date.apply(
                lambda x: vwap_map.get(pd.Timestamp(x).date() - pd.Timedelta(days=1), np.nan)
            )
        
        return result_df
