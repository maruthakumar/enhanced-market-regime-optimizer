"""
ATR Indicators Module

This module implements ATR (Average True Range) indicators for volatility analysis
across multiple timeframes.

Features:
- ATR calculation
- ATR percentile
- ATR/EMA ratio
- ATR-based volatility regime detection
- ATR expansion/contraction detection
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="ATRIndicators", category="atr")
class ATRIndicators(FeatureBase):
    """
    ATR indicators for volatility analysis.
    
    This class calculates ATR (Average True Range) and derived indicators
    for volatility analysis across multiple timeframes.
    """
    
    def __init__(self, config=None):
        """
        Initialize ATR indicators.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.atr_period = int(self.config.get('atr_period', 14))
        self.percentile_lookback = int(self.config.get('percentile_lookback', 100))
        self.ema_periods = self.config.get('ema_periods', [20, 50, 100])
        self.timeframes = self.config.get('timeframes', ['5m', '10m', '15m'])
        
        logger.info(f"Initialized ATR indicators with period {self.atr_period} "
                   f"and percentile lookback {self.percentile_lookback}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate ATR indicators.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - timeframes (list): Override default timeframes
                - atr_period (int): Override default ATR period
                - percentile_lookback (int): Override default percentile lookback
            
        Returns:
            pd.DataFrame: Data with calculated ATR indicators
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Override defaults with kwargs if provided
        timeframes = kwargs.get('timeframes', self.timeframes)
        atr_period = kwargs.get('atr_period', self.atr_period)
        percentile_lookback = kwargs.get('percentile_lookback', self.percentile_lookback)
        
        # Calculate ATR for each timeframe
        for timeframe in timeframes:
            # Check if required columns exist
            required_cols = ['High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing required columns for ATR calculation: {missing_cols}")
                continue
            
            # Calculate True Range
            df[f'TR_{timeframe}'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            
            # Calculate ATR
            df[f'ATR_{timeframe}'] = df[f'TR_{timeframe}'].rolling(window=atr_period).mean()
            
            # Calculate ATR percentile
            df[f'ATR_{timeframe}_Percentile'] = df[f'ATR_{timeframe}'].rolling(window=percentile_lookback).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
            
            # Calculate ATR/Close ratio (volatility relative to price)
            df[f'ATR_{timeframe}_Ratio'] = df[f'ATR_{timeframe}'] / df['Close']
            
            # Calculate ATR/Close ratio percentile
            df[f'ATR_{timeframe}_Ratio_Percentile'] = df[f'ATR_{timeframe}_Ratio'].rolling(window=percentile_lookback).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
            
            # Calculate ATR EMA
            for period in self.ema_periods:
                df[f'ATR_{timeframe}_EMA{period}'] = df[f'ATR_{timeframe}'].ewm(span=period, adjust=False).mean()
            
            # Calculate ATR expansion/contraction
            if len(self.ema_periods) >= 2:
                # Sort periods to ensure correct order
                sorted_periods = sorted(self.ema_periods)
                
                # Compare shortest EMA to longest EMA
                short_ema = f'ATR_{timeframe}_EMA{sorted_periods[0]}'
                long_ema = f'ATR_{timeframe}_EMA{sorted_periods[-1]}'
                
                # ATR expansion: short EMA > long EMA
                # ATR contraction: short EMA < long EMA
                df[f'ATR_{timeframe}_Expansion'] = 0
                df.loc[df[short_ema] > df[long_ema], f'ATR_{timeframe}_Expansion'] = 1
                df.loc[df[short_ema] < df[long_ema], f'ATR_{timeframe}_Expansion'] = -1
            
            # Calculate volatility regime based on ATR percentile
            df[f'Volatility_Regime_{timeframe}'] = 'Normal_Vol'
            df.loc[df[f'ATR_{timeframe}_Percentile'] < 0.2, f'Volatility_Regime_{timeframe}'] = 'Very_Low_Vol'
            df.loc[(df[f'ATR_{timeframe}_Percentile'] >= 0.2) & 
                   (df[f'ATR_{timeframe}_Percentile'] < 0.4), f'Volatility_Regime_{timeframe}'] = 'Low_Vol'
            df.loc[(df[f'ATR_{timeframe}_Percentile'] >= 0.6) & 
                   (df[f'ATR_{timeframe}_Percentile'] < 0.8), f'Volatility_Regime_{timeframe}'] = 'High_Vol'
            df.loc[df[f'ATR_{timeframe}_Percentile'] >= 0.8, f'Volatility_Regime_{timeframe}'] = 'Extreme_Vol'
        
        logger.info(f"Calculated ATR indicators for {len(timeframes)} timeframes")
        
        return df
    
    @cache_result
    def calculate_atr_bands(self, data_frame, timeframe='15m', multipliers=[1.0, 1.5, 2.0, 2.5, 3.0]):
        """
        Calculate ATR bands (similar to Keltner Channels).
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframe (str): Timeframe to use
            multipliers (list): List of ATR multipliers for bands
            
        Returns:
            pd.DataFrame: Data with ATR bands
        """
        df = data_frame.copy()
        
        # Check if ATR column exists
        atr_col = f'ATR_{timeframe}'
        if atr_col not in df.columns:
            logger.warning(f"ATR column {atr_col} not found in data")
            # Calculate ATR
            df = self.calculate_features(df, timeframes=[timeframe])
        
        # Check if required columns exist
        if 'Close' not in df.columns or atr_col not in df.columns:
            logger.warning("Missing required columns for ATR bands calculation")
            return df
        
        # Calculate middle band (20-period EMA)
        middle_band = f'ATR_Band_Middle_{timeframe}'
        df[middle_band] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Calculate upper and lower bands for each multiplier
        for mult in multipliers:
            upper_band = f'ATR_Band_Upper_{timeframe}_{mult:.1f}'
            lower_band = f'ATR_Band_Lower_{timeframe}_{mult:.1f}'
            
            df[upper_band] = df[middle_band] + mult * df[atr_col]
            df[lower_band] = df[middle_band] - mult * df[atr_col]
        
        # Calculate price position relative to bands
        df[f'ATR_Band_Position_{timeframe}'] = (df['Close'] - df[middle_band]) / df[atr_col]
        
        logger.info(f"Calculated ATR bands for timeframe {timeframe}")
        
        return df
    
    @cache_result
    def calculate_volatility_breakout(self, data_frame, timeframe='15m', lookback=20, threshold=2.0):
        """
        Calculate volatility breakout signals.
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframe (str): Timeframe to use
            lookback (int): Lookback period for volatility comparison
            threshold (float): Threshold for volatility breakout
            
        Returns:
            pd.DataFrame: Data with volatility breakout signals
        """
        df = data_frame.copy()
        
        # Check if ATR column exists
        atr_col = f'ATR_{timeframe}'
        if atr_col not in df.columns:
            logger.warning(f"ATR column {atr_col} not found in data")
            # Calculate ATR
            df = self.calculate_features(df, timeframes=[timeframe])
        
        # Check if required columns exist
        if atr_col not in df.columns:
            logger.warning("Missing required columns for volatility breakout calculation")
            return df
        
        # Calculate average ATR over lookback period
        df[f'ATR_{timeframe}_Avg_{lookback}'] = df[atr_col].rolling(window=lookback).mean()
        
        # Calculate ATR ratio (current ATR / average ATR)
        df[f'ATR_{timeframe}_Ratio_{lookback}'] = df[atr_col] / df[f'ATR_{timeframe}_Avg_{lookback}']
        
        # Detect volatility breakout
        df[f'Volatility_Breakout_{timeframe}'] = 0
        df.loc[df[f'ATR_{timeframe}_Ratio_{lookback}'] > threshold, f'Volatility_Breakout_{timeframe}'] = 1
        
        logger.info(f"Calculated volatility breakout signals for timeframe {timeframe}")
        
        return df
