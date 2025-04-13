"""
IV Indicators Module

This module implements Implied Volatility (IV) indicators for options analysis
with DTE-wise bucketing and skew analysis.

Features:
- IV percentile calculation
- IV skew analysis
- DTE-wise IV percentile
- IV term structure analysis
- IV-based market regime detection
"""

import pandas as pd
import numpy as np
import logging
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="IVIndicators", category="iv")
class IVIndicators(FeatureBase):
    """
    IV indicators for options analysis.
    
    This class calculates IV percentile, IV skew, and DTE-wise IV metrics
    for options trading analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize IV indicators.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        self.lookback_period = int(self.config.get('lookback_period', 60))
        self.dte_buckets = self.config.get('dte_buckets', [0, 7, 14, 30, 60, 90])
        self.use_skew = self.config.get('use_skew', True)
        self.use_term_structure = self.config.get('use_term_structure', True)
        
        logger.info(f"Initialized IV indicators with lookback period {self.lookback_period} "
                   f"and DTE buckets {self.dte_buckets}")
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate IV indicators.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - iv_column (str): Column name for IV
                - dte_column (str): Column name for DTE
                - call_iv_column (str): Column name for Call IV
                - put_iv_column (str): Column name for Put IV
            
        Returns:
            pd.DataFrame: Data with calculated IV indicators
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Get column names from kwargs or use defaults
        iv_column = kwargs.get('iv_column', 'ATM_IV')
        dte_column = kwargs.get('dte_column', 'DTE')
        call_iv_column = kwargs.get('call_iv_column', 'ATM_Call_IV')
        put_iv_column = kwargs.get('put_iv_column', 'ATM_Put_IV')
        
        # Check if IV column exists
        if iv_column not in df.columns:
            logger.warning(f"IV column {iv_column} not found in data")
            # Try to calculate IV from call and put IV if available
            if call_iv_column in df.columns and put_iv_column in df.columns:
                logger.info(f"Calculating {iv_column} from {call_iv_column} and {put_iv_column}")
                df[iv_column] = (df[call_iv_column] + df[put_iv_column]) / 2
            else:
                logger.error(f"Cannot calculate IV indicators without IV data")
                return df
        
        # Calculate IV percentile
        df['IV_Percentile'] = df[iv_column].rolling(window=self.lookback_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        ).fillna(0.5)
        
        # Calculate IV regime based on percentile
        df['IV_Regime'] = 'Normal_IV'
        df.loc[df['IV_Percentile'] < 0.2, 'IV_Regime'] = 'Very_Low_IV'
        df.loc[(df['IV_Percentile'] >= 0.2) & (df['IV_Percentile'] < 0.4), 'IV_Regime'] = 'Low_IV'
        df.loc[(df['IV_Percentile'] >= 0.6) & (df['IV_Percentile'] < 0.8), 'IV_Regime'] = 'High_IV'
        df.loc[df['IV_Percentile'] >= 0.8, 'IV_Regime'] = 'Extreme_IV'
        
        # Calculate IV skew if enabled and required columns exist
        if self.use_skew and call_iv_column in df.columns and put_iv_column in df.columns:
            # IV skew (put IV / call IV)
            df['IV_Skew'] = df[put_iv_column] / df[call_iv_column]
            
            # Handle infinity and NaN
            df['IV_Skew'] = df['IV_Skew'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            
            # IV skew percentile
            df['IV_Skew_Percentile'] = df['IV_Skew'].rolling(window=self.lookback_period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
            
            # IV skew regime
            df['IV_Skew_Regime'] = 'Normal_Skew'
            df.loc[df['IV_Skew_Percentile'] < 0.2, 'IV_Skew_Regime'] = 'Very_Low_Skew'
            df.loc[(df['IV_Skew_Percentile'] >= 0.2) & (df['IV_Skew_Percentile'] < 0.4), 'IV_Skew_Regime'] = 'Low_Skew'
            df.loc[(df['IV_Skew_Percentile'] >= 0.6) & (df['IV_Skew_Percentile'] < 0.8), 'IV_Skew_Regime'] = 'High_Skew'
            df.loc[df['IV_Skew_Percentile'] >= 0.8, 'IV_Skew_Regime'] = 'Extreme_Skew'
        
        # Calculate DTE-wise IV percentile if DTE column exists
        if dte_column in df.columns:
            # Create DTE bucket column
            df['DTE_Bucket'] = pd.cut(df[dte_column], bins=self.dte_buckets, right=False)
            
            # Calculate IV percentile for each DTE bucket
            for i in range(len(self.dte_buckets)-1):
                lower = self.dte_buckets[i]
                upper = self.dte_buckets[i+1]
                bucket_name = f"DTE_{lower}_{upper}"
                
                # Filter data for this bucket
                bucket_mask = (df[dte_column] >= lower) & (df[dte_column] < upper)
                
                if bucket_mask.any():
                    # Calculate IV percentile for this bucket
                    bucket_data = df.loc[bucket_mask, iv_column]
                    
                    # Use expanding window for percentile calculation within each bucket
                    df.loc[bucket_mask, f'IV_Percentile_{bucket_name}'] = bucket_data.expanding().apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    ).fillna(0.5)
                    
                    # Calculate IV regime for this bucket
                    df.loc[bucket_mask, f'IV_Regime_{bucket_name}'] = 'Normal_IV'
                    df.loc[bucket_mask & (df[f'IV_Percentile_{bucket_name}'] < 0.2), f'IV_Regime_{bucket_name}'] = 'Very_Low_IV'
                    df.loc[bucket_mask & (df[f'IV_Percentile_{bucket_name}'] >= 0.2) & 
                           (df[f'IV_Percentile_{bucket_name}'] < 0.4), f'IV_Regime_{bucket_name}'] = 'Low_IV'
                    df.loc[bucket_mask & (df[f'IV_Percentile_{bucket_name}'] >= 0.6) & 
                           (df[f'IV_Percentile_{bucket_name}'] < 0.8), f'IV_Regime_{bucket_name}'] = 'High_IV'
                    df.loc[bucket_mask & (df[f'IV_Percentile_{bucket_name}'] >= 0.8), f'IV_Regime_{bucket_name}'] = 'Extreme_IV'
        
        # Calculate IV term structure if enabled and DTE column exists
        if self.use_term_structure and dte_column in df.columns:
            # Group by date and calculate IV term structure
            if 'Date' in df.columns:
                # Create a copy of the dataframe with only the necessary columns
                term_df = df[['Date', dte_column, iv_column]].copy()
                
                # Group by date
                date_groups = term_df.groupby('Date')
                
                # Calculate IV term structure for each date
                for date, group in date_groups:
                    if len(group) > 1:
                        # Sort by DTE
                        sorted_group = group.sort_values(by=dte_column)
                        
                        # Calculate IV slope (simple linear regression)
                        x = sorted_group[dte_column].values
                        y = sorted_group[iv_column].values
                        
                        if len(x) > 1:
                            # Calculate slope using numpy's polyfit
                            slope, _ = np.polyfit(x, y, 1)
                            
                            # Store slope in the dataframe
                            df.loc[df['Date'] == date, 'IV_Term_Slope'] = slope
                
                # Calculate IV term structure regime
                if 'IV_Term_Slope' in df.columns:
                    # Calculate percentile of slope
                    df['IV_Term_Slope_Percentile'] = df['IV_Term_Slope'].rolling(window=self.lookback_period).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    ).fillna(0.5)
                    
                    # Define term structure regime
                    df['IV_Term_Structure_Regime'] = 'Normal_Term'
                    df.loc[df['IV_Term_Slope_Percentile'] < 0.2, 'IV_Term_Structure_Regime'] = 'Very_Flat_Term'
                    df.loc[(df['IV_Term_Slope_Percentile'] >= 0.2) & 
                           (df['IV_Term_Slope_Percentile'] < 0.4), 'IV_Term_Structure_Regime'] = 'Flat_Term'
                    df.loc[(df['IV_Term_Slope_Percentile'] >= 0.6) & 
                           (df['IV_Term_Slope_Percentile'] < 0.8), 'IV_Term_Structure_Regime'] = 'Steep_Term'
                    df.loc[df['IV_Term_Slope_Percentile'] >= 0.8, 'IV_Term_Structure_Regime'] = 'Very_Steep_Term'
        
        logger.info(f"Calculated IV indicators")
        
        return df
    
    @cache_result
    def calculate_iv_surface(self, data_frame, strike_columns=None, dte_column='DTE'):
        """
        Calculate IV surface metrics.
        
        Args:
            data_frame (pd.DataFrame): Input data
            strike_columns (list): List of columns containing IV at different strikes
            dte_column (str): Column name for DTE
            
        Returns:
            pd.DataFrame: Data with IV surface metrics
        """
        df = data_frame.copy()
        
        # Check if DTE column exists
        if dte_column not in df.columns:
            logger.warning(f"DTE column {dte_column} not found in data")
            return df
        
        # If no strike IV columns provided, try to find them
        if strike_columns is None:
            strike_columns = [col for col in df.columns if 'Strike' in col and 'IV' in col]
            
            if not strike_columns:
                logger.warning("No strike IV columns found")
                return df
        
        # Calculate IV smile (curvature of IV across strikes)
        if len(strike_columns) >= 3:
            # Assume strike columns are ordered from lowest to highest strike
            low_strike = strike_columns[0]
            mid_strike = strike_columns[len(strike_columns) // 2]
            high_strike = strike_columns[-1]
            
            # Calculate smile curvature (higher value = more pronounced smile)
            df['IV_Smile_Curvature'] = (df[low_strike] + df[high_strike] - 2 * df[mid_strike]) / df[mid_strike]
            
            # Calculate IV smile percentile
            df['IV_Smile_Percentile'] = df['IV_Smile_Curvature'].rolling(window=self.lookback_period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            ).fillna(0.5)
            
            # Define IV smile regime
            df['IV_Smile_Regime'] = 'Normal_Smile'
            df.loc[df['IV_Smile_Percentile'] < 0.2, 'IV_Smile_Regime'] = 'Very_Flat_Smile'
            df.loc[(df['IV_Smile_Percentile'] >= 0.2) & 
                   (df['IV_Smile_Percentile'] < 0.4), 'IV_Smile_Regime'] = 'Flat_Smile'
            df.loc[(df['IV_Smile_Percentile'] >= 0.6) & 
                   (df['IV_Smile_Percentile'] < 0.8), 'IV_Smile_Regime'] = 'Pronounced_Smile'
            df.loc[df['IV_Smile_Percentile'] >= 0.8, 'IV_Smile_Regime'] = 'Extreme_Smile'
        
        # Calculate IV surface metrics if we have multiple DTE values
        if df[dte_column].nunique() > 1:
            # Group by date
            if 'Date' in df.columns:
                # Create a copy of the dataframe with only the necessary columns
                surface_df = df[['Date', dte_column] + strike_columns].copy()
                
                # Group by date
                date_groups = surface_df.groupby('Date')
                
                # Calculate IV surface metrics for each date
                for date, group in date_groups:
                    if len(group) > 1:
                        # Calculate average IV for each DTE
                        group['Avg_IV'] = group[strike_columns].mean(axis=1)
                        
                        # Sort by DTE
                        sorted_group = group.sort_values(by=dte_column)
                        
                        if len(sorted_group) > 1:
                            # Calculate term structure slope
                            x = sorted_group[dte_column].values
                            y = sorted_group['Avg_IV'].values
                            
                            # Calculate slope using numpy's polyfit
                            slope, _ = np.polyfit(x, y, 1)
                            
                            # Store slope in the dataframe
                            df.loc[df['Date'] == date, 'IV_Surface_Slope'] = slope
                
                # Calculate IV surface regime
                if 'IV_Surface_Slope' in df.columns:
                    # Calculate percentile of slope
                    df['IV_Surface_Slope_Percentile'] = df['IV_Surface_Slope'].rolling(window=self.lookback_period).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    ).fillna(0.5)
                    
                    # Define surface regime
                    df['IV_Surface_Regime'] = 'Normal_Surface'
                    df.loc[df['IV_Surface_Slope_Percentile'] < 0.2, 'IV_Surface_Regime'] = 'Very_Flat_Surface'
                    df.loc[(df['IV_Surface_Slope_Percentile'] >= 0.2) & 
                           (df['IV_Surface_Slope_Percentile'] < 0.4), 'IV_Surface_Regime'] = 'Flat_Surface'
                    df.loc[(df['IV_Surface_Slope_Percentile'] >= 0.6) & 
                           (df['IV_Surface_Slope_Percentile'] < 0.8), 'IV_Surface_Regime'] = 'Steep_Surface'
                    df.loc[df['IV_Surface_Slope_Percentile'] >= 0.8, 'IV_Surface_Regime'] = 'Very_Steep_Surface'
        
        logger.info(f"Calculated IV surface metrics")
        
        return df
