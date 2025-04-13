"""
Volume Indicators Module

This module implements volume analysis and patterns for market participation
and liquidity condition detection.

Features:
- Volume moving averages
- Volume ratio (current/average)
- Volume percentile
- Volume trend detection
- Volume-price relationship
- Volume-based signals
- Volume profile analysis
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="VolumeIndicators", category="volume")
class VolumeIndicators(FeatureBase):
    """
    Volume indicators for market participation and liquidity analysis.
    
    This class calculates volume patterns and trends to identify market
    participation and liquidity conditions.
    """
    
    def __init__(self, config=None):
        """
        Initialize volume indicators.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.ma_periods = self.config.get('ma_periods', [20, 50, 100])
        self.lookback_period = int(self.config.get('lookback_period', 60))
        self.use_profile = self.config.get('use_profile', True)
        self.profile_bins = int(self.config.get('profile_bins', 20))
        self.timeframes = self.config.get('timeframes', ['5m', '10m', '15m'])
        
        logger.info(f"Initialized volume indicators with MA periods {self.ma_periods} "
                   f"and lookback period {self.lookback_period}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate volume indicators.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - timeframes (list): Override default timeframes
                - volume_column (str): Column name for volume
                - price_column (str): Column name for price
            
        Returns:
            pd.DataFrame: Data with calculated volume indicators
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Override defaults with kwargs if provided
        timeframes = kwargs.get('timeframes', self.timeframes)
        volume_column = kwargs.get('volume_column', 'Volume')
        price_column = kwargs.get('price_column', 'Close')
        
        # Check if required columns exist
        if volume_column not in df.columns:
            logger.warning(f"Volume column {volume_column} not found in data")
            return df
        
        # Calculate volume indicators for each timeframe
        for timeframe in timeframes:
            # Calculate volume moving averages
            for period in self.ma_periods:
                df[f'Volume_MA{period}_{timeframe}'] = df[volume_column].rolling(window=period).mean()
            
            # Calculate volume ratio (current volume / moving average)
            for period in self.ma_periods:
                df[f'Volume_Ratio{period}_{timeframe}'] = df[volume_column] / df[f'Volume_MA{period}_{timeframe}']
                
                # Handle infinity and NaN
                df[f'Volume_Ratio{period}_{timeframe}'] = df[f'Volume_Ratio{period}_{timeframe}'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # Calculate volume percentile
            df[f'Volume_Percentile_{timeframe}'] = df[volume_column].rolling(window=self.lookback_period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
            
            # Calculate volume trend
            df[f'Volume_Trend_{timeframe}'] = 0  # 0 = neutral, 1 = increasing, -1 = decreasing
            
            # Use shortest MA period for trend detection
            shortest_ma = min(self.ma_periods)
            df.loc[df[f'Volume_Ratio{shortest_ma}_{timeframe}'] > 1.2, f'Volume_Trend_{timeframe}'] = 1
            df.loc[df[f'Volume_Ratio{shortest_ma}_{timeframe}'] < 0.8, f'Volume_Trend_{timeframe}'] = -1
            
            # Calculate volume-price relationship if price column exists
            if price_column in df.columns:
                # Calculate price change
                df[f'Price_Change_{timeframe}'] = df[price_column].pct_change()
                
                # Calculate volume-price relationship
                # 1 = bullish (price up, volume up)
                # -1 = bearish (price down, volume up)
                # 0 = neutral or conflicting signals
                df[f'Volume_Price_Relationship_{timeframe}'] = 0
                
                # Bullish: price up, volume up
                bullish = (df[f'Price_Change_{timeframe}'] > 0) & (df[f'Volume_Trend_{timeframe}'] > 0)
                df.loc[bullish, f'Volume_Price_Relationship_{timeframe}'] = 1
                
                # Bearish: price down, volume up
                bearish = (df[f'Price_Change_{timeframe}'] < 0) & (df[f'Volume_Trend_{timeframe}'] > 0)
                df.loc[bearish, f'Volume_Price_Relationship_{timeframe}'] = -1
                
                # Calculate volume-based signals
                df[f'Volume_Signal_{timeframe}'] = 0
                
                # Bullish signal: price up, volume up
                df.loc[bullish, f'Volume_Signal_{timeframe}'] = 1
                
                # Bearish signal: price down, volume up
                df.loc[bearish, f'Volume_Signal_{timeframe}'] = -1
                
                # Potential reversal: price up, volume down (exhaustion)
                exhaustion_up = (df[f'Price_Change_{timeframe}'] > 0) & (df[f'Volume_Trend_{timeframe}'] < 0)
                df.loc[exhaustion_up, f'Volume_Signal_{timeframe}'] = -0.5
                
                # Potential reversal: price down, volume down (exhaustion)
                exhaustion_down = (df[f'Price_Change_{timeframe}'] < 0) & (df[f'Volume_Trend_{timeframe}'] < 0)
                df.loc[exhaustion_down, f'Volume_Signal_{timeframe}'] = 0.5
            
            # Calculate liquidity regime based on volume percentile
            df[f'Liquidity_Regime_{timeframe}'] = 'Normal_Liquidity'
            df.loc[df[f'Volume_Percentile_{timeframe}'] < 0.1, f'Liquidity_Regime_{timeframe}'] = 'Drying_Liquidity'
            df.loc[(df[f'Volume_Percentile_{timeframe}'] >= 0.1) & 
                   (df[f'Volume_Percentile_{timeframe}'] < 0.3), f'Liquidity_Regime_{timeframe}'] = 'Low_Liquidity'
            df.loc[df[f'Volume_Percentile_{timeframe}'] > 0.7, f'Liquidity_Regime_{timeframe}'] = 'High_Liquidity'
        
        # Calculate volume profile if enabled and price column exists
        if self.use_profile and price_column in df.columns:
            # Calculate volume profile
            self._calculate_volume_profile(df, volume_column, price_column)
        
        logger.info(f"Calculated volume indicators for {len(timeframes)} timeframes")
        
        return df
    
    def _calculate_volume_profile(self, df, volume_column, price_column):
        """
        Calculate volume profile.
        
        Args:
            df (pd.DataFrame): Input data
            volume_column (str): Column name for volume
            price_column (str): Column name for price
        """
        # Check if we have enough data
        if len(df) < 20:
            logger.warning("Not enough data for volume profile calculation")
            return
        
        # Get price range
        price_min = df[price_column].min()
        price_max = df[price_column].max()
        
        # Create price bins
        price_bins = np.linspace(price_min, price_max, self.profile_bins + 1)
        
        # Create bin labels
        bin_labels = [f'Bin_{i}' for i in range(self.profile_bins)]
        
        # Cut prices into bins
        df['Price_Bin'] = pd.cut(df[price_column], bins=price_bins, labels=bin_labels, include_lowest=True)
        
        # Group by bin and sum volume
        volume_profile = df.groupby('Price_Bin')[volume_column].sum()
        
        # Calculate Point of Control (POC) - price level with highest volume
        poc_bin = volume_profile.idxmax()
        poc_index = bin_labels.index(poc_bin)
        poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
        
        # Store POC in dataframe
        df['Volume_POC'] = poc_price
        
        # Calculate Value Area (70% of volume)
        total_volume = volume_profile.sum()
        value_area_volume = total_volume * 0.7
        
        # Sort bins by volume in descending order
        sorted_bins = volume_profile.sort_values(ascending=False)
        
        # Calculate cumulative volume
        cum_volume = sorted_bins.cumsum()
        
        # Get bins in value area
        value_area_bins = cum_volume[cum_volume <= value_area_volume].index.tolist()
        
        # Get price levels for value area
        value_area_indices = [bin_labels.index(bin_name) for bin_name in value_area_bins]
        value_area_prices = [(price_bins[i], price_bins[i + 1]) for i in value_area_indices]
        
        # Calculate Value Area High (VAH) and Value Area Low (VAL)
        if value_area_prices:
            vah = max([p[1] for p in value_area_prices])
            val = min([p[0] for p in value_area_prices])
            
            # Store VAH and VAL in dataframe
            df['Volume_VAH'] = vah
            df['Volume_VAL'] = val
        
        # Store volume profile in dataframe
        for bin_name, bin_volume in volume_profile.items():
            df[f'Volume_Profile_{bin_name}'] = bin_volume
        
        # Clean up temporary column
        df = df.drop('Price_Bin', axis=1)
    
    @cache_result
    def calculate_volume_delta(self, data_frame, timeframe='15m', volume_column='Volume', price_column='Close'):
        """
        Calculate volume delta (buying vs. selling pressure).
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframe (str): Timeframe to use
            volume_column (str): Column name for volume
            price_column (str): Column name for price
            
        Returns:
            pd.DataFrame: Data with volume delta
        """
        df = data_frame.copy()
        
        # Check if required columns exist
        if volume_column not in df.columns or price_column not in df.columns:
            logger.warning(f"Missing required columns for volume delta calculation")
            return df
        
        # Calculate price change
        df['Price_Change'] = df[price_column].pct_change()
        
        # Calculate volume delta
        df[f'Volume_Delta_{timeframe}'] = df[volume_column] * df['Price_Change'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        # Calculate cumulative volume delta
        df[f'Cumulative_Volume_Delta_{timeframe}'] = df[f'Volume_Delta_{timeframe}'].cumsum()
        
        # Calculate volume delta moving average
        df[f'Volume_Delta_MA_{timeframe}'] = df[f'Volume_Delta_{timeframe}'].rolling(window=20).mean()
        
        # Calculate volume delta trend
        df[f'Volume_Delta_Trend_{timeframe}'] = 0
        df.loc[df[f'Volume_Delta_MA_{timeframe}'] > 0, f'Volume_Delta_Trend_{timeframe}'] = 1
        df.loc[df[f'Volume_Delta_MA_{timeframe}'] < 0, f'Volume_Delta_Trend_{timeframe}'] = -1
        
        # Clean up temporary column
        df = df.drop('Price_Change', axis=1)
        
        logger.info(f"Calculated volume delta for timeframe {timeframe}")
        
        return df
    
    @cache_result
    def calculate_volume_divergence(self, data_frame, timeframe='15m', volume_column='Volume', price_column='Close'):
        """
        Calculate volume-price divergence.
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframe (str): Timeframe to use
            volume_column (str): Column name for volume
            price_column (str): Column name for price
            
        Returns:
            pd.DataFrame: Data with volume-price divergence
        """
        df = data_frame.copy()
        
        # Check if required columns exist
        if volume_column not in df.columns or price_column not in df.columns:
            logger.warning(f"Missing required columns for volume divergence calculation")
            return df
        
        # Calculate price and volume moving averages
        df[f'Price_MA_{timeframe}'] = df[price_column].rolling(window=20).mean()
        df[f'Volume_MA_{timeframe}'] = df[volume_column].rolling(window=20).mean()
        
        # Calculate price and volume trends
        df[f'Price_Trend_{timeframe}'] = (df[price_column] > df[f'Price_MA_{timeframe}']).astype(int) * 2 - 1
        df[f'Volume_Trend_{timeframe}'] = (df[volume_column] > df[f'Volume_MA_{timeframe}']).astype(int) * 2 - 1
        
        # Calculate volume-price divergence
        df[f'Volume_Price_Divergence_{timeframe}'] = 0
        
        # Bullish divergence: price down, volume up
        bullish_div = (df[f'Price_Trend_{timeframe}'] < 0) & (df[f'Volume_Trend_{timeframe}'] > 0)
        df.loc[bullish_div, f'Volume_Price_Divergence_{timeframe}'] = 1
        
        # Bearish divergence: price up, volume down
        bearish_div = (df[f'Price_Trend_{timeframe}'] > 0) & (df[f'Volume_Trend_{timeframe}'] < 0)
        df.loc[bearish_div, f'Volume_Price_Divergence_{timeframe}'] = -1
        
        logger.info(f"Calculated volume-price divergence for timeframe {timeframe}")
        
        return df
