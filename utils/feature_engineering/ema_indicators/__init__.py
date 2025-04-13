"""
EMA Indicators Module

This module implements EMA (Exponential Moving Average) indicators for ATM straddle and ATM CE/PE
across multiple timeframes (5m, 10m, 15m).

Features:
- EMA 20, 100, 200 calculations
- Multi-timeframe support (5m, 10m, 15m)
- EMA crossover detection
- EMA slope and momentum analysis
- EMA-based trend strength indicators
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="EMAIndicators", category="ema")
class EMAIndicators(FeatureBase):
    """
    EMA indicators for ATM straddle and ATM CE/PE across multiple timeframes.
    
    This class calculates EMA 20, 100, 200 on ATM straddle and ATM CE/PE
    for 5m, 10m, and 15m timeframes, along with derived indicators.
    """
    
    def __init__(self, config=None):
        """
        Initialize EMA indicators.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.ema_periods = self.config.get('ema_periods', [20, 100, 200])
        self.timeframes = self.config.get('timeframes', ['5m', '10m', '15m'])
        self.price_columns = self.config.get('price_columns', 
                                           ['ATM_Straddle_Premium', 'ATM_CE_Premium', 'ATM_PE_Premium'])
        self.use_slope = self.config.get('use_slope', True)
        self.use_crossover = self.config.get('use_crossover', True)
        self.use_alignment = self.config.get('use_alignment', True)
        
        logger.info(f"Initialized EMA indicators with periods {self.ema_periods} "
                   f"for timeframes {self.timeframes}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate EMA indicators.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - timeframes (list): Override default timeframes
                - ema_periods (list): Override default EMA periods
                - price_columns (list): Override default price columns
            
        Returns:
            pd.DataFrame: Data with calculated EMA indicators
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Override defaults with kwargs if provided
        timeframes = kwargs.get('timeframes', self.timeframes)
        ema_periods = kwargs.get('ema_periods', self.ema_periods)
        price_columns = kwargs.get('price_columns', self.price_columns)
        
        # Calculate EMAs for each timeframe, price column, and period
        for timeframe in timeframes:
            for price_col in price_columns:
                # Check if the price column exists
                if price_col not in df.columns:
                    logger.warning(f"Price column {price_col} not found in data")
                    continue
                
                # Calculate EMAs for each period
                for period in ema_periods:
                    ema_col = f"{price_col}_{timeframe}_EMA{period}"
                    df[ema_col] = df[price_col].ewm(span=period, adjust=False).mean()
                    
                    # Calculate EMA position (price relative to EMA)
                    if self.use_slope:
                        position_col = f"{price_col}_{timeframe}_EMA{period}_Position"
                        df[position_col] = (df[price_col] - df[ema_col]) / df[ema_col]
                    
                    # Calculate EMA slope
                    if self.use_slope:
                        slope_col = f"{price_col}_{timeframe}_EMA{period}_Slope"
                        df[slope_col] = df[ema_col].pct_change(5)
                
                # Calculate EMA crossovers if we have at least 2 periods
                if self.use_crossover and len(ema_periods) >= 2:
                    for i in range(len(ema_periods)-1):
                        short_period = ema_periods[i]
                        long_period = ema_periods[i+1]
                        
                        short_ema = f"{price_col}_{timeframe}_EMA{short_period}"
                        long_ema = f"{price_col}_{timeframe}_EMA{long_period}"
                        
                        crossover_col = f"{price_col}_{timeframe}_EMA{short_period}_{long_period}_Crossover"
                        
                        # 1 for bullish crossover (short above long), -1 for bearish, 0 for no crossover
                        df[crossover_col] = 0
                        
                        # Current crossover state
                        df.loc[df[short_ema] > df[long_ema], crossover_col] = 1
                        df.loc[df[short_ema] < df[long_ema], crossover_col] = -1
                
                # Calculate EMA alignment if we have at least 3 periods
                if self.use_alignment and len(ema_periods) >= 3:
                    alignment_col = f"{price_col}_{timeframe}_EMA_Alignment"
                    
                    # Initialize alignment column
                    df[alignment_col] = 0
                    
                    # Sort periods to ensure correct order
                    sorted_periods = sorted(ema_periods)
                    
                    # Get EMA column names in order
                    ema_cols = [f"{price_col}_{timeframe}_EMA{period}" for period in sorted_periods]
                    
                    # Check for bullish alignment (shorter EMAs above longer EMAs)
                    bullish_alignment = True
                    for i in range(len(ema_cols)-1):
                        if not (df[ema_cols[i]] > df[ema_cols[i+1]]).all():
                            bullish_alignment = False
                            break
                    
                    if bullish_alignment:
                        df[alignment_col] = 1
                    
                    # Check for bearish alignment (shorter EMAs below longer EMAs)
                    bearish_alignment = True
                    for i in range(len(ema_cols)-1):
                        if not (df[ema_cols[i]] < df[ema_cols[i+1]]).all():
                            bearish_alignment = False
                            break
                    
                    if bearish_alignment:
                        df[alignment_col] = -1
        
        logger.info(f"Calculated EMA indicators for {len(timeframes)} timeframes, "
                   f"{len(price_columns)} price columns, and {len(ema_periods)} periods")
        
        return df
    
    @cache_result
    def calculate_ema_trend_strength(self, data_frame, timeframe='15m', price_col='ATM_Straddle_Premium'):
        """
        Calculate EMA-based trend strength.
        
        Args:
            data_frame (pd.DataFrame): Input data
            timeframe (str): Timeframe to use
            price_col (str): Price column to use
            
        Returns:
            pd.DataFrame: Data with trend strength indicator
        """
        df = data_frame.copy()
        
        # Check if we have the necessary EMA columns
        ema_cols = [f"{price_col}_{timeframe}_EMA{period}" for period in self.ema_periods]
        missing_cols = [col for col in ema_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing EMA columns: {missing_cols}")
            # Calculate missing EMAs
            df = self.calculate_features(df, 
                                        timeframes=[timeframe], 
                                        price_columns=[price_col],
                                        ema_periods=self.ema_periods)
        
        # Calculate trend strength based on price vs EMAs
        strength_col = f"{price_col}_{timeframe}_Trend_Strength"
        
        # Initialize with zeros
        df[strength_col] = 0
        
        # For each EMA, add +1 if price is above, -1 if below
        for period in self.ema_periods:
            ema_col = f"{price_col}_{timeframe}_EMA{period}"
            df.loc[df[price_col] > df[ema_col], strength_col] += 1
            df.loc[df[price_col] < df[ema_col], strength_col] -= 1
        
        # Normalize to range [-1, 1]
        df[strength_col] = df[strength_col] / len(self.ema_periods)
        
        logger.info(f"Calculated EMA trend strength for {timeframe} {price_col}")
        
        return df
