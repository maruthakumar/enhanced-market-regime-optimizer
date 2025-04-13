"""
Trending OI with PA (Open Interest with Price Action) Module

This module implements trending Open Interest with Price Action analysis
for market regime detection and trading signal generation.

Features:
- OI trend detection (increasing/decreasing patterns)
- OI accumulation detection
- OI at key strike levels
- Price momentum relative to OI changes
- Breakout/breakdown confirmation with OI
- Support/resistance tests with volume and OI confirmation
- Divergence/convergence between OI and price
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="TrendingOIPA", category="oi_pa")
class TrendingOIPA(FeatureBase):
    """
    Trending Open Interest with Price Action analysis.
    
    This class analyzes the relationship between Open Interest trends and Price Action
    to identify potential trading opportunities and market regimes.
    """
    
    def __init__(self, config=None):
        """
        Initialize Trending OI with PA analysis.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.oi_lookback = int(self.config.get('oi_lookback', 10))
        self.price_lookback = int(self.config.get('price_lookback', 5))
        self.divergence_threshold = float(self.config.get('divergence_threshold', 0.1))
        self.accumulation_threshold = float(self.config.get('accumulation_threshold', 0.2))
        self.use_percentile = self.config.get('use_percentile', True)
        self.percentile_window = int(self.config.get('percentile_window', 20))
        self.timeframes = self.config.get('timeframes', ['5m', '10m', '15m'])
        
        logger.info(f"Initialized Trending OI with PA analysis with lookback periods "
                   f"OI: {self.oi_lookback}, Price: {self.price_lookback}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate Trending OI with PA features.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - timeframes (list): Override default timeframes
                - oi_column (str): Column name for Open Interest
                - price_column (str): Column name for Price
            
        Returns:
            pd.DataFrame: Data with calculated Trending OI with PA features
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Override defaults with kwargs if provided
        timeframes = kwargs.get('timeframes', self.timeframes)
        oi_column = kwargs.get('oi_column', 'OI')
        price_column = kwargs.get('price_column', 'Close')
        
        # Check if required columns exist
        if oi_column not in df.columns:
            logger.warning(f"OI column {oi_column} not found in data")
            return df
        
        if price_column not in df.columns:
            logger.warning(f"Price column {price_column} not found in data")
            return df
        
        # Calculate OI trend
        df[f'{oi_column}_Change'] = df[oi_column].pct_change()
        df[f'{oi_column}_Change_5D'] = df[oi_column].pct_change(5)
        df[f'{oi_column}_Change_10D'] = df[oi_column].pct_change(10)
        
        # Calculate OI moving averages
        df[f'{oi_column}_MA5'] = df[oi_column].rolling(window=5).mean()
        df[f'{oi_column}_MA10'] = df[oi_column].rolling(window=10).mean()
        df[f'{oi_column}_MA20'] = df[oi_column].rolling(window=20).mean()
        
        # Calculate OI trend indicators
        df['OI_Trend'] = 0  # 0 = neutral, 1 = increasing, -1 = decreasing
        df.loc[df[f'{oi_column}_MA5'] > df[f'{oi_column}_MA20'], 'OI_Trend'] = 1
        df.loc[df[f'{oi_column}_MA5'] < df[f'{oi_column}_MA20'], 'OI_Trend'] = -1
        
        # Calculate OI percentile if enabled
        if self.use_percentile:
            df['OI_Percentile'] = df[oi_column].rolling(window=self.percentile_window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
        
        # Calculate OI accumulation
        df['OI_Accumulation'] = 0
        
        # OI increasing for consecutive days indicates accumulation
        oi_changes = df[f'{oi_column}_Change'].rolling(window=self.oi_lookback).apply(
            lambda x: (x > 0).sum() / len(x)
        ).fillna(0.5)
        
        df.loc[oi_changes > (0.5 + self.accumulation_threshold), 'OI_Accumulation'] = 1
        df.loc[oi_changes < (0.5 - self.accumulation_threshold), 'OI_Accumulation'] = -1
        
        # Calculate price momentum
        df[f'{price_column}_Change'] = df[price_column].pct_change()
        df[f'{price_column}_Change_5D'] = df[price_column].pct_change(5)
        
        # Calculate price momentum indicators
        df['Price_Momentum'] = 0  # 0 = neutral, 1 = bullish, -1 = bearish
        df.loc[df[f'{price_column}_Change_5D'] > 0, 'Price_Momentum'] = 1
        df.loc[df[f'{price_column}_Change_5D'] < 0, 'Price_Momentum'] = -1
        
        # Calculate OI-Price divergence
        df['OI_Price_Divergence'] = 0
        
        # Bullish divergence: Price down, OI up
        bullish_div = (df[f'{price_column}_Change_5D'] < 0) & (df[f'{oi_column}_Change_5D'] > 0)
        df.loc[bullish_div, 'OI_Price_Divergence'] = 1
        
        # Bearish divergence: Price up, OI down
        bearish_div = (df[f'{price_column}_Change_5D'] > 0) & (df[f'{oi_column}_Change_5D'] < 0)
        df.loc[bearish_div, 'OI_Price_Divergence'] = -1
        
        # Calculate OI-Price confirmation
        df['OI_Price_Confirmation'] = 0
        
        # Bullish confirmation: Price up, OI up
        bullish_conf = (df[f'{price_column}_Change_5D'] > 0) & (df[f'{oi_column}_Change_5D'] > 0)
        df.loc[bullish_conf, 'OI_Price_Confirmation'] = 1
        
        # Bearish confirmation: Price down, OI down
        bearish_conf = (df[f'{price_column}_Change_5D'] < 0) & (df[f'{oi_column}_Change_5D'] < 0)
        df.loc[bearish_conf, 'OI_Price_Confirmation'] = -1
        
        # Calculate OI-based support/resistance
        df['OI_Support_Resistance'] = 0
        
        # Support: Price near recent low, high OI
        support_cond = (df[price_column] <= df[price_column].rolling(20).min() * 1.02) & \
                      (df[oi_column] >= df[oi_column].rolling(20).quantile(0.8))
        df.loc[support_cond, 'OI_Support_Resistance'] = 1
        
        # Resistance: Price near recent high, high OI
        resistance_cond = (df[price_column] >= df[price_column].rolling(20).max() * 0.98) & \
                         (df[oi_column] >= df[oi_column].rolling(20).quantile(0.8))
        df.loc[resistance_cond, 'OI_Support_Resistance'] = -1
        
        # Calculate combined OI-PA indicator
        df['OI_PA_Signal'] = df['OI_Trend'] + df['Price_Momentum'] + df['OI_Price_Divergence']
        
        # Normalize to range [-1, 1]
        df['OI_PA_Signal'] = df['OI_PA_Signal'] / 3
        
        # Calculate OI-PA regime
        df['OI_PA_Regime'] = 'Neutral'
        df.loc[df['OI_PA_Signal'] >= 0.5, 'OI_PA_Regime'] = 'Bullish'
        df.loc[df['OI_PA_Signal'] <= -0.5, 'OI_PA_Regime'] = 'Bearish'
        df.loc[(df['OI_PA_Signal'] > 0) & (df['OI_PA_Signal'] < 0.5), 'OI_PA_Regime'] = 'Weak_Bullish'
        df.loc[(df['OI_PA_Signal'] < 0) & (df['OI_PA_Signal'] > -0.5), 'OI_PA_Regime'] = 'Weak_Bearish'
        
        logger.info(f"Calculated Trending OI with PA features")
        
        return df
    
    @cache_result
    def calculate_oi_strike_analysis(self, data_frame, strike_oi_columns=None):
        """
        Calculate OI analysis at key strike levels.
        
        Args:
            data_frame (pd.DataFrame): Input data
            strike_oi_columns (list): List of columns containing OI at different strikes
            
        Returns:
            pd.DataFrame: Data with OI strike analysis
        """
        df = data_frame.copy()
        
        # If no strike OI columns provided, try to find them
        if strike_oi_columns is None:
            strike_oi_columns = [col for col in df.columns if 'Strike' in col and 'OI' in col]
            
            if not strike_oi_columns:
                logger.warning("No strike OI columns found")
                return df
        
        # Calculate total OI across all strikes
        df['Total_Strike_OI'] = df[strike_oi_columns].sum(axis=1)
        
        # Calculate OI concentration
        for col in strike_oi_columns:
            df[f'{col}_Concentration'] = df[col] / df['Total_Strike_OI']
        
        # Find strike with maximum OI
        df['Max_OI_Strike'] = df[strike_oi_columns].idxmax(axis=1)
        df['Max_OI_Concentration'] = df[strike_oi_columns].max(axis=1) / df['Total_Strike_OI']
        
        # Calculate OI dispersion (lower value = more concentrated)
        df['OI_Dispersion'] = 1 - df[strike_oi_columns].apply(
            lambda x: (x / x.sum()).max(), axis=1
        )
        
        # Calculate magnetic price levels based on OI concentration
        df['OI_Magnetic_Level'] = 0
        
        # If we have strike price information
        strike_price_cols = [col.replace('OI', 'Price') for col in strike_oi_columns 
                           if col.replace('OI', 'Price') in df.columns]
        
        if strike_price_cols:
            # Calculate weighted average strike price based on OI
            weights = df[strike_oi_columns].values
            prices = df[strike_price_cols].values
            
            # Avoid division by zero
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            
            # Calculate weighted average
            weighted_prices = (weights * prices) / row_sums
            df['OI_Weighted_Strike'] = weighted_prices.sum(axis=1)
            
            # Current price column (assuming 'Close' or similar exists)
            price_col = 'Close'
            if price_col in df.columns:
                # Calculate distance from current price to weighted strike
                df['Distance_To_Magnetic_Level'] = (df['OI_Weighted_Strike'] - df[price_col]) / df[price_col]
        
        logger.info(f"Calculated OI strike analysis")
        
        return df
    
    @cache_result
    def calculate_oi_expiration_effect(self, data_frame, dte_column='DTE'):
        """
        Calculate OI expiration effect.
        
        Args:
            data_frame (pd.DataFrame): Input data
            dte_column (str): Days to expiration column
            
        Returns:
            pd.DataFrame: Data with OI expiration effect
        """
        df = data_frame.copy()
        
        # Check if DTE column exists
        if dte_column not in df.columns:
            logger.warning(f"DTE column {dte_column} not found in data")
            return df
        
        # OI column
        oi_column = 'OI'
        if oi_column not in df.columns:
            logger.warning(f"OI column {oi_column} not found in data")
            return df
        
        # Calculate OI change by DTE
        df['OI_Expiration_Effect'] = 0
        
        # Group by DTE and calculate average OI change
        if len(df) > 0:
            dte_groups = df.groupby(dte_column)
            
            for dte, group in dte_groups:
                if len(group) > 1:
                    avg_change = group[oi_column].pct_change().mean()
                    df.loc[df[dte_column] == dte, 'OI_Expiration_Effect'] = avg_change
        
        # Calculate expiration pressure (higher when close to expiration with high OI)
        df['Expiration_Pressure'] = 0
        
        # Only calculate for rows with DTE <= 5
        mask = df[dte_column] <= 5
        if mask.any():
            # Normalize OI to 0-1 range
            oi_normalized = (df.loc[mask, oi_column] - df.loc[mask, oi_column].min()) / \
                           (df.loc[mask, oi_column].max() - df.loc[mask, oi_column].min() + 1e-10)
            
            # Pressure increases as DTE decreases
            dte_factor = 1 - (df.loc[mask, dte_column] / 5)
            
            # Combine factors
            df.loc[mask, 'Expiration_Pressure'] = oi_normalized * dte_factor
        
        logger.info(f"Calculated OI expiration effect")
        
        return df
