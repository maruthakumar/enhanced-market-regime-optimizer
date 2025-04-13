"""
Premium Percentile Indicators Module

This module implements premium percentile indicators for options analysis,
with DTE-wise bucketing and trend detection.

Features:
- ATM straddle premium percentile based on DTE
- ATM CE/PE premium percentile based on DTE
- Premium ratio analysis
- Premium trend detection
- Premium-based market signals
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="PremiumPercentileIndicators", category="premium")
class PremiumPercentileIndicators(FeatureBase):
    """
    Premium percentile indicators for options analysis.
    
    This class calculates premium percentiles for ATM straddle and ATM CE/PE,
    with DTE-wise bucketing and trend detection.
    """
    
    def __init__(self, config=None):
        """
        Initialize premium percentile indicators.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.lookback_period = int(self.config.get('lookback_period', 60))
        self.dte_buckets = self.config.get('dte_buckets', [0, 7, 14, 30, 60, 90])
        self.use_ratio = self.config.get('use_ratio', True)
        self.use_trend = self.config.get('use_trend', True)
        
        logger.info(f"Initialized premium percentile indicators with lookback period {self.lookback_period} "
                   f"and DTE buckets {self.dte_buckets}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate premium percentile indicators.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - straddle_column (str): Column name for ATM straddle premium
                - call_column (str): Column name for ATM call premium
                - put_column (str): Column name for ATM put premium
                - dte_column (str): Column name for DTE
            
        Returns:
            pd.DataFrame: Data with calculated premium percentile indicators
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Get column names from kwargs or use defaults
        straddle_column = kwargs.get('straddle_column', 'ATM_Straddle_Premium')
        call_column = kwargs.get('call_column', 'ATM_Call_Premium')
        put_column = kwargs.get('put_column', 'ATM_Put_Premium')
        dte_column = kwargs.get('dte_column', 'DTE')
        
        # Check if required columns exist
        missing_columns = []
        for col in [straddle_column, call_column, put_column, dte_column]:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # If we're missing all premium columns, we can't calculate percentiles
            if all(col in missing_columns for col in [straddle_column, call_column, put_column]):
                logger.error("No premium columns found, cannot calculate percentiles")
                return df
        
        # Calculate premium percentiles for each available premium column
        premium_columns = []
        if straddle_column in df.columns:
            premium_columns.append(straddle_column)
        if call_column in df.columns:
            premium_columns.append(call_column)
        if put_column in df.columns:
            premium_columns.append(put_column)
        
        # Calculate DTE-wise premium percentiles if DTE column exists
        if dte_column in df.columns:
            # Create DTE bucket column
            df['DTE_Bucket'] = pd.cut(df[dte_column], bins=self.dte_buckets, right=False)
            
            # Calculate premium percentile for each DTE bucket and premium column
            for premium_col in premium_columns:
                for i in range(len(self.dte_buckets)-1):
                    lower = self.dte_buckets[i]
                    upper = self.dte_buckets[i+1]
                    bucket_name = f"DTE_{lower}_{upper}"
                    
                    # Filter data for this bucket
                    bucket_mask = (df[dte_column] >= lower) & (df[dte_column] < upper)
                    
                    if bucket_mask.any():
                        # Calculate premium percentile for this bucket
                        bucket_data = df.loc[bucket_mask, premium_col]
                        
                        # Use expanding window for percentile calculation within each bucket
                        df.loc[bucket_mask, f'{premium_col}_Percentile_{bucket_name}'] = bucket_data.expanding().apply(
                            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                        ).fillna(0.5)
                        
                        # Calculate premium regime for this bucket
                        df.loc[bucket_mask, f'{premium_col}_Regime_{bucket_name}'] = 'Normal_Premium'
                        df.loc[bucket_mask & (df[f'{premium_col}_Percentile_{bucket_name}'] < 0.2), 
                               f'{premium_col}_Regime_{bucket_name}'] = 'Very_Low_Premium'
                        df.loc[bucket_mask & (df[f'{premium_col}_Percentile_{bucket_name}'] >= 0.2) & 
                               (df[f'{premium_col}_Percentile_{bucket_name}'] < 0.4), 
                               f'{premium_col}_Regime_{bucket_name}'] = 'Low_Premium'
                        df.loc[bucket_mask & (df[f'{premium_col}_Percentile_{bucket_name}'] >= 0.6) & 
                               (df[f'{premium_col}_Percentile_{bucket_name}'] < 0.8), 
                               f'{premium_col}_Regime_{bucket_name}'] = 'High_Premium'
                        df.loc[bucket_mask & (df[f'{premium_col}_Percentile_{bucket_name}'] >= 0.8), 
                               f'{premium_col}_Regime_{bucket_name}'] = 'Extreme_Premium'
        else:
            # Calculate overall premium percentiles if DTE column doesn't exist
            for premium_col in premium_columns:
                df[f'{premium_col}_Percentile'] = df[premium_col].rolling(window=self.lookback_period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                ).fillna(0.5)
                
                # Calculate premium regime
                df[f'{premium_col}_Regime'] = 'Normal_Premium'
                df.loc[df[f'{premium_col}_Percentile'] < 0.2, f'{premium_col}_Regime'] = 'Very_Low_Premium'
                df.loc[(df[f'{premium_col}_Percentile'] >= 0.2) & 
                       (df[f'{premium_col}_Percentile'] < 0.4), f'{premium_col}_Regime'] = 'Low_Premium'
                df.loc[(df[f'{premium_col}_Percentile'] >= 0.6) & 
                       (df[f'{premium_col}_Percentile'] < 0.8), f'{premium_col}_Regime'] = 'High_Premium'
                df.loc[df[f'{premium_col}_Percentile'] >= 0.8, f'{premium_col}_Regime'] = 'Extreme_Premium'
        
        # Calculate premium ratios if enabled and required columns exist
        if self.use_ratio and call_column in df.columns and put_column in df.columns:
            # Put/Call premium ratio
            df['Put_Call_Premium_Ratio'] = df[put_column] / df[call_column]
            
            # Handle infinity and NaN
            df['Put_Call_Premium_Ratio'] = df['Put_Call_Premium_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # Put/Call premium ratio percentile
            df['Put_Call_Premium_Ratio_Percentile'] = df['Put_Call_Premium_Ratio'].rolling(window=self.lookback_period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
            
            # Put/Call premium ratio regime
            df['Put_Call_Premium_Ratio_Regime'] = 'Normal_Ratio'
            df.loc[df['Put_Call_Premium_Ratio_Percentile'] < 0.2, 'Put_Call_Premium_Ratio_Regime'] = 'Very_Low_Ratio'
            df.loc[(df['Put_Call_Premium_Ratio_Percentile'] >= 0.2) & 
                   (df['Put_Call_Premium_Ratio_Percentile'] < 0.4), 'Put_Call_Premium_Ratio_Regime'] = 'Low_Ratio'
            df.loc[(df['Put_Call_Premium_Ratio_Percentile'] >= 0.6) & 
                   (df['Put_Call_Premium_Ratio_Percentile'] < 0.8), 'Put_Call_Premium_Ratio_Regime'] = 'High_Ratio'
            df.loc[df['Put_Call_Premium_Ratio_Percentile'] >= 0.8, 'Put_Call_Premium_Ratio_Regime'] = 'Extreme_Ratio'
        
        # Calculate premium trends if enabled
        if self.use_trend:
            for premium_col in premium_columns:
                # Calculate 5-day change
                df[f'{premium_col}_Change_5D'] = df[premium_col].pct_change(5)
                
                # Calculate trend indicator
                df[f'{premium_col}_Trend'] = 0  # 0 = neutral, 1 = increasing, -1 = decreasing
                df.loc[df[f'{premium_col}_Change_5D'] > 0.05, f'{premium_col}_Trend'] = 1
                df.loc[df[f'{premium_col}_Change_5D'] < -0.05, f'{premium_col}_Trend'] = -1
                
                # Calculate trend strength
                df[f'{premium_col}_Trend_Strength'] = df[f'{premium_col}_Change_5D'].abs()
        
        logger.info(f"Calculated premium percentile indicators")
        
        return df
    
    @cache_result
    def calculate_premium_surface(self, data_frame, dte_column='DTE', date_column='Date'):
        """
        Calculate premium surface metrics.
        
        Args:
            data_frame (pd.DataFrame): Input data
            dte_column (str): Column name for DTE
            date_column (str): Column name for date
            
        Returns:
            pd.DataFrame: Data with premium surface metrics
        """
        df = data_frame.copy()
        
        # Check if required columns exist
        if dte_column not in df.columns or date_column not in df.columns:
            logger.warning(f"Missing required columns for premium surface calculation")
            return df
        
        # Get premium columns
        premium_columns = [col for col in df.columns if 'Premium' in col and not any(x in col for x in ['Percentile', 'Regime', 'Ratio', 'Change', 'Trend'])]
        
        if not premium_columns:
            logger.warning("No premium columns found for surface calculation")
            return df
        
        # Calculate premium surface for each premium column
        for premium_col in premium_columns:
            # Group by date
            date_groups = df.groupby(date_column)
            
            # Calculate premium surface for each date
            for date, group in date_groups:
                if len(group) > 1:
                    # Sort by DTE
                    sorted_group = group.sort_values(by=dte_column)
                    
                    if len(sorted_group) > 1:
                        # Calculate term structure slope
                        x = sorted_group[dte_column].values
                        y = sorted_group[premium_col].values
                        
                        # Calculate slope using numpy's polyfit
                        slope, _ = np.polyfit(x, y, 1)
                        
                        # Store slope in the dataframe
                        df.loc[df[date_column] == date, f'{premium_col}_Term_Slope'] = slope
            
            # Calculate term structure percentile
            if f'{premium_col}_Term_Slope' in df.columns:
                df[f'{premium_col}_Term_Slope_Percentile'] = df[f'{premium_col}_Term_Slope'].rolling(window=self.lookback_period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                ).fillna(0.5)
                
                # Define term structure regime
                df[f'{premium_col}_Term_Structure_Regime'] = 'Normal_Term'
                df.loc[df[f'{premium_col}_Term_Slope_Percentile'] < 0.2, f'{premium_col}_Term_Structure_Regime'] = 'Very_Flat_Term'
                df.loc[(df[f'{premium_col}_Term_Slope_Percentile'] >= 0.2) & 
                       (df[f'{premium_col}_Term_Slope_Percentile'] < 0.4), f'{premium_col}_Term_Structure_Regime'] = 'Flat_Term'
                df.loc[(df[f'{premium_col}_Term_Slope_Percentile'] >= 0.6) & 
                       (df[f'{premium_col}_Term_Slope_Percentile'] < 0.8), f'{premium_col}_Term_Structure_Regime'] = 'Steep_Term'
                df.loc[df[f'{premium_col}_Term_Slope_Percentile'] >= 0.8, f'{premium_col}_Term_Structure_Regime'] = 'Very_Steep_Term'
        
        logger.info(f"Calculated premium surface metrics")
        
        return df
