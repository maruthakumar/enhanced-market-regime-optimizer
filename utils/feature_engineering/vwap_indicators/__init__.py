"""
VWAP Indicators Module

This module implements Volume Weighted Average Price (VWAP) indicators
for intraday trading and multi-day trend analysis.

Features:
- VWAP calculation
- Previous day's VWAP
- VWAP bands
- Price relative to VWAP
- VWAP-based signals
- Multi-timeframe VWAP analysis
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="VWAPIndicators", category="vwap")
class VWAPIndicators(FeatureBase):
    """
    VWAP indicators for intraday trading and multi-day trend analysis.
    
    This class calculates VWAP, VWAP bands, and VWAP-based signals
    for trading analysis across multiple timeframes.
    """
    
    def __init__(self, config=None):
        """
        Initialize VWAP indicators.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.band_multipliers = self.config.get('band_multipliers', [1.0, 1.5, 2.0, 2.5, 3.0])
        self.use_prev_day = self.config.get('use_prev_day', True)
        self.timeframes = self.config.get('timeframes', ['5m', '10m', '15m'])
        
        logger.info(f"Initialized VWAP indicators with band multipliers {self.band_multipliers}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate VWAP indicators.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - timeframes (list): Override default timeframes
                - price_column (str): Column name for price
                - volume_column (str): Column name for volume
                - date_column (str): Column name for date
            
        Returns:
            pd.DataFrame: Data with calculated VWAP indicators
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Override defaults with kwargs if provided
        timeframes = kwargs.get('timeframes', self.timeframes)
        price_column = kwargs.get('price_column', 'Close')
        volume_column = kwargs.get('volume_column', 'Volume')
        date_column = kwargs.get('date_column', 'Date')
        
        # Check if required columns exist
        if price_column not in df.columns:
            logger.warning(f"Price column {price_column} not found in data")
            return df
        
        if volume_column not in df.columns:
            logger.warning(f"Volume column {volume_column} not found in data")
            return df
        
        # Calculate VWAP for each timeframe
        for timeframe in timeframes:
            # Calculate typical price (TP)
            if 'High' in df.columns and 'Low' in df.columns:
                df['TP'] = (df['High'] + df['Low'] + df[price_column]) / 3
            else:
                df['TP'] = df[price_column]
            
            # Calculate cumulative TP * Volume
            df['TP_Volume'] = df['TP'] * df[volume_column]
            
            # If date column exists, group by date to reset VWAP calculation each day
            if date_column in df.columns:
                # Group by date
                date_groups = df.groupby(date_column)
                
                # Calculate VWAP for each date
                for date, group in date_groups:
                    # Get indices for this date
                    date_indices = group.index
                    
                    # Calculate cumulative sum for this date
                    cum_tp_volume = group['TP_Volume'].cumsum()
                    cum_volume = group[volume_column].cumsum()
                    
                    # Calculate VWAP
                    vwap = cum_tp_volume / cum_volume
                    
                    # Store in dataframe
                    df.loc[date_indices, f'VWAP_{timeframe}'] = vwap
                    
                    # Calculate standard deviation of price from VWAP
                    price_deviation = (group[price_column] - vwap) ** 2
                    std_dev = np.sqrt(price_deviation.cumsum() / cum_volume)
                    
                    # Calculate VWAP bands
                    for mult in self.band_multipliers:
                        df.loc[date_indices, f'VWAP_Band_Upper_{timeframe}_{mult:.1f}'] = vwap + mult * std_dev
                        df.loc[date_indices, f'VWAP_Band_Lower_{timeframe}_{mult:.1f}'] = vwap - mult * std_dev
                    
                    # Calculate price position relative to VWAP
                    df.loc[date_indices, f'VWAP_Position_{timeframe}'] = (group[price_column] - vwap) / vwap
            else:
                # Calculate VWAP without date grouping
                cum_tp_volume = df['TP_Volume'].cumsum()
                cum_volume = df[volume_column].cumsum()
                
                # Calculate VWAP
                df[f'VWAP_{timeframe}'] = cum_tp_volume / cum_volume
                
                # Calculate standard deviation of price from VWAP
                price_deviation = (df[price_column] - df[f'VWAP_{timeframe}']) ** 2
                std_dev = np.sqrt(price_deviation.cumsum() / cum_volume)
                
                # Calculate VWAP bands
                for mult in self.band_multipliers:
                    df[f'VWAP_Band_Upper_{timeframe}_{mult:.1f}'] = df[f'VWAP_{timeframe}'] + mult * std_dev
                    df[f'VWAP_Band_Lower_{timeframe}_{mult:.1f}'] = df[f'VWAP_{timeframe}'] - mult * std_dev
                
                # Calculate price position relative to VWAP
                df[f'VWAP_Position_{timeframe}'] = (df[price_column] - df[f'VWAP_{timeframe}']) / df[f'VWAP_{timeframe}']
            
            # Calculate VWAP-based signals
            df[f'VWAP_Signal_{timeframe}'] = 0  # 0 = neutral, 1 = bullish, -1 = bearish
            df.loc[df[price_column] > df[f'VWAP_{timeframe}'], f'VWAP_Signal_{timeframe}'] = 1
            df.loc[df[price_column] < df[f'VWAP_{timeframe}'], f'VWAP_Signal_{timeframe}'] = -1
            
            # Calculate VWAP trend
            df[f'VWAP_Trend_{timeframe}'] = df[f'VWAP_{timeframe}'].pct_change(5).apply(
                lambda x: 1 if x > 0.001 else (-1 if x < -0.001 else 0)
            )
        
        # Calculate previous day's VWAP if enabled and date column exists
        if self.use_prev_day and date_column in df.columns:
            # Group by date
            date_groups = df.groupby(date_column)
            
            # Get unique dates
            unique_dates = df[date_column].unique()
            
            # Calculate previous day's VWAP for each date
            for i, date in enumerate(unique_dates):
                if i > 0:
                    prev_date = unique_dates[i-1]
                    
                    # Get previous day's data
                    prev_day_data = df[df[date_column] == prev_date]
                    
                    if len(prev_day_data) > 0:
                        # Calculate previous day's VWAP
                        prev_day_tp_volume = prev_day_data['TP'] * prev_day_data[volume_column]
                        prev_day_vwap = prev_day_tp_volume.sum() / prev_day_data[volume_column].sum()
                        
                        # Store in dataframe
                        df.loc[df[date_column] == date, 'Prev_Day_VWAP'] = prev_day_vwap
                        
                        # Calculate price position relative to previous day's VWAP
                        df.loc[df[date_column] == date, 'Prev_Day_VWAP_Position'] = (
                            df.loc[df[date_column] == date, price_column] - prev_day_vwap
                        ) / prev_day_vwap
        
        # Clean up temporary columns
        df = df.drop(['TP', 'TP_Volume'], axis=1)
        
        logger.info(f"Calculated VWAP indicators for {len(timeframes)} timeframes")
        
        return df
    
    @cache_result
    def calculate_vwap_crossovers(self, data_frame, timeframe='15m', price_column='Close'):
        """
        Calculate VWAP crossover signals.
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframe (str): Timeframe to use
            price_column (str): Column name for price
            
        Returns:
            pd.DataFrame: Data with VWAP crossover signals
        """
        df = data_frame.copy()
        
        # Check if VWAP column exists
        vwap_col = f'VWAP_{timeframe}'
        if vwap_col not in df.columns:
            logger.warning(f"VWAP column {vwap_col} not found in data")
            return df
        
        # Check if price column exists
        if price_column not in df.columns:
            logger.warning(f"Price column {price_column} not found in data")
            return df
        
        # Calculate price position relative to VWAP
        df[f'Price_Above_VWAP_{timeframe}'] = df[price_column] > df[vwap_col]
        
        # Calculate crossover signals
        df[f'VWAP_Crossover_{timeframe}'] = 0
        
        # Bullish crossover: Price crosses above VWAP
        bullish_crossover = (~df[f'Price_Above_VWAP_{timeframe}'].shift(1)) & df[f'Price_Above_VWAP_{timeframe}']
        df.loc[bullish_crossover, f'VWAP_Crossover_{timeframe}'] = 1
        
        # Bearish crossover: Price crosses below VWAP
        bearish_crossover = df[f'Price_Above_VWAP_{timeframe}'].shift(1) & (~df[f'Price_Above_VWAP_{timeframe}'])
        df.loc[bearish_crossover, f'VWAP_Crossover_{timeframe}'] = -1
        
        # Calculate distance from VWAP at crossover
        df[f'VWAP_Crossover_Distance_{timeframe}'] = 0
        df.loc[df[f'VWAP_Crossover_{timeframe}'] != 0, f'VWAP_Crossover_Distance_{timeframe}'] = (
            abs(df[price_column] - df[vwap_col]) / df[vwap_col]
        )
        
        logger.info(f"Calculated VWAP crossover signals for timeframe {timeframe}")
        
        return df
    
    @cache_result
    def calculate_multi_timeframe_vwap(self, data_frame, timeframes=None, price_column='Close'):
        """
        Calculate multi-timeframe VWAP analysis.
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframes (list): List of timeframes to analyze
            price_column (str): Column name for price
            
        Returns:
            pd.DataFrame: Data with multi-timeframe VWAP analysis
        """
        df = data_frame.copy()
        
        # Use default timeframes if none provided
        if timeframes is None:
            timeframes = self.timeframes
        
        # Check if price column exists
        if price_column not in df.columns:
            logger.warning(f"Price column {price_column} not found in data")
            return df
        
        # Check if VWAP columns exist
        vwap_cols = [f'VWAP_{tf}' for tf in timeframes]
        missing_cols = [col for col in vwap_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing VWAP columns: {missing_cols}")
            return df
        
        # Calculate multi-timeframe VWAP alignment
        df['VWAP_Alignment'] = 0
        
        # Count how many timeframes have price above VWAP
        above_count = sum([(df[price_column] > df[f'VWAP_{tf}']).astype(int) for tf in timeframes])
        
        # Calculate alignment score (-1 to 1 scale)
        df['VWAP_Alignment'] = (2 * above_count / len(timeframes)) - 1
        
        # Calculate VWAP alignment regime
        df['VWAP_Alignment_Regime'] = 'Neutral'
        df.loc[df['VWAP_Alignment'] >= 0.6, 'VWAP_Alignment_Regime'] = 'Bullish'
        df.loc[df['VWAP_Alignment'] <= -0.6, 'VWAP_Alignment_Regime'] = 'Bearish'
        df.loc[(df['VWAP_Alignment'] > 0) & (df['VWAP_Alignment'] < 0.6), 'VWAP_Alignment_Regime'] = 'Weak_Bullish'
        df.loc[(df['VWAP_Alignment'] < 0) & (df['VWAP_Alignment'] > -0.6), 'VWAP_Alignment_Regime'] = 'Weak_Bearish'
        
        logger.info(f"Calculated multi-timeframe VWAP analysis for {len(timeframes)} timeframes")
        
        return df
