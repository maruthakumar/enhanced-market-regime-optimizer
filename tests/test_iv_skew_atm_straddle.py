"""
Script to test IV skew and ATM straddle analysis for market regime testing
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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

class IVSkewATMStraddleTester:
    """
    Class to test IV skew and ATM straddle analysis for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the IV skew and ATM straddle tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/iv_skew_atm_straddle')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # IV skew parameters
        self.otm_call_delta = self.config.get('otm_call_delta', 0.25)
        self.otm_put_delta = self.config.get('otm_put_delta', 0.25)
        self.lookback_period = self.config.get('lookback_period', 20)
        self.percentile_window = self.config.get('percentile_window', 252)  # Approximately 1 year of trading days
        
        logger.info(f"Initialized IVSkewATMStraddleTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"IV skew parameters: otm_call_delta={self.otm_call_delta}, otm_put_delta={self.otm_put_delta}, lookback_period={self.lookback_period}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for IV skew and ATM straddle testing")
        
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
    
    def calculate_iv_skew_indicators(self, df):
        """
        Calculate IV skew and ATM straddle indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with IV skew and ATM straddle indicators
        """
        logger.info("Calculating IV skew and ATM straddle indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_columns = ['IV', 'Strike', 'Option_Type', 'Underlying_Price']
        
        # If Underlying_Price doesn't exist but Price does, use Price
        if 'Underlying_Price' not in result_df.columns and 'Price' in result_df.columns:
            logger.info("Using 'Price' column as 'Underlying_Price'")
            result_df['Underlying_Price'] = result_df['Price']
        
        # Check if required columns exist after adjustments
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns after adjustments: {missing_columns}")
            
            # If IV is missing but we have option prices, try to calculate synthetic IV
            if 'IV' in missing_columns and 'Price' in result_df.columns and 'Strike' in result_df.columns:
                logger.warning("IV column missing, creating synthetic IV data")
                
                # Create synthetic IV based on moneyness (distance from ATM)
                if 'Underlying_Price' in result_df.columns:
                    # Calculate moneyness
                    result_df['Moneyness'] = result_df['Strike'] / result_df['Underlying_Price']
                    
                    # Create synthetic IV with a smile curve
                    # ATM options have lower IV, OTM options have higher IV
                    result_df['IV'] = 0.2 + 0.1 * (result_df['Moneyness'] - 1)**2
                    
                    # Add some randomness
                    np.random.seed(42)  # For reproducibility
                    result_df['IV'] = result_df['IV'] * (1 + np.random.normal(0, 0.05, len(result_df)))
                    
                    # Ensure IV is positive
                    result_df['IV'] = result_df['IV'].clip(lower=0.05)
                    
                    logger.info("Created synthetic IV data based on moneyness")
                    missing_columns.remove('IV')
        
        # If still missing required columns, return the original dataframe
        if missing_columns:
            logger.error(f"Still missing required columns: {missing_columns}")
            return result_df
        
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
        
        try:
            # Process data by datetime to calculate IV skew and ATM straddle
            if 'datetime' in result_df.columns:
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize result columns
                result_df['ATM_IV'] = np.nan
                result_df['OTM_Call_IV'] = np.nan
                result_df['OTM_Put_IV'] = np.nan
                result_df['IV_Skew'] = np.nan
                result_df['IV_Skew_Percentile'] = np.nan
                result_df['ATM_Straddle_Premium'] = np.nan
                result_df['ATM_Straddle_Premium_Percentile'] = np.nan
                
                # Process each datetime group
                for dt, group in datetime_groups:
                    # Find ATM strike
                    underlying_price = group['Underlying_Price'].iloc[0]
                    strikes = group['Strike'].unique()
                    atm_strike = strikes[np.abs(strikes - underlying_price).argmin()]
                    
                    # Get ATM options
                    atm_call = group[(group['Strike'] == atm_strike) & (group['Option_Type'].str.lower() == 'call')]
                    atm_put = group[(group['Strike'] == atm_strike) & (group['Option_Type'].str.lower() == 'put')]
                    
                    # Calculate ATM IV
                    if len(atm_call) > 0 and len(atm_put) > 0:
                        atm_call_iv = atm_call['IV'].iloc[0]
                        atm_put_iv = atm_put['IV'].iloc[0]
                        atm_iv = (atm_call_iv + atm_put_iv) / 2
                    elif len(atm_call) > 0:
                        atm_iv = atm_call['IV'].iloc[0]
                    elif len(atm_put) > 0:
                        atm_iv = atm_put['IV'].iloc[0]
                    else:
                        atm_iv = np.nan
                    
                    # Find OTM strikes
                    otm_call_strike = None
                    otm_put_strike = None
                    
                    # If we have Delta column, use it to find OTM options
                    if 'Delta' in group.columns:
                        # Find OTM call with delta closest to target
                        otm_calls = group[(group['Strike'] > atm_strike) & (group['Option_Type'].str.lower() == 'call')]
                        if len(otm_calls) > 0:
                            otm_calls['Delta_Diff'] = abs(otm_calls['Delta'] - self.otm_call_delta)
                            otm_call = otm_calls.loc[otm_calls['Delta_Diff'].idxmin()]
                            otm_call_strike = otm_call['Strike']
                        
                        # Find OTM put with delta closest to target
                        otm_puts = group[(group['Strike'] < atm_strike) & (group['Option_Type'].str.lower() == 'put')]
                        if len(otm_puts) > 0:
                            otm_puts['Delta_Diff'] = abs(abs(otm_puts['Delta']) - self.otm_put_delta)
                            otm_put = otm_puts.loc[otm_puts['Delta_Diff'].idxmin()]
                            otm_put_strike = otm_put['Strike']
                    else:
                        # If no Delta column, estimate OTM strikes based on percentage OTM
                        otm_call_strike = atm_strike * 1.05  # 5% OTM call
                        otm_put_strike = atm_strike * 0.95   # 5% OTM put
                        
                        # Find closest available strikes
                        if len(strikes) > 0:
                            otm_call_strike = strikes[np.abs(strikes - otm_call_strike).argmin()]
                            otm_put_strike = strikes[np.abs(strikes - otm_put_strike).argmin()]
                    
                    # Get OTM options
                    otm_call = group[(group['Strike'] == otm_call_strike) & (group['Option_Type'].str.lower() == 'call')] if otm_call_strike is not None else pd.DataFrame()
                    otm_put = group[(group['Strike'] == otm_put_strike) & (group['Option_Type'].str.lower() == 'put')] if otm_put_strike is not None else pd.DataFrame()
                    
                    # Calculate OTM IVs
                    otm_call_iv = otm_call['IV'].iloc[0] if len(otm_call) > 0 else np.nan
                    otm_put_iv = otm_put['IV'].iloc[0] if len(otm_put) > 0 else np.nan
                    
                    # Calculate IV skew
                    if not np.isnan(otm_put_iv) and not np.isnan(otm_call_iv):
                        iv_skew = otm_put_iv - otm_call_iv
                    else:
                        iv_skew = np.nan
                    
                    # Calculate ATM straddle premium
                    if len(atm_call) > 0 and len(atm_put) > 0 and 'Price' in atm_call.columns and 'Price' in atm_put.columns:
                        atm_call_price = atm_call['Price'].iloc[0]
                        atm_put_price = atm_put['Price'].iloc[0]
                        atm_straddle_premium = atm_call_price + atm_put_price
                    else:
                        atm_straddle_premium = np.nan
                    
                    # Update result dataframe
                    result_df.loc[result_df['datetime'] == dt, 'ATM_IV'] = atm_iv
                    result_df.loc[result_df['datetime'] == dt, 'OTM_Call_IV'] = otm_call_iv
                    result_df.loc[result_df['datetime'] == dt, 'OTM_Put_IV'] = otm_put_iv
                    result_df.loc[result_df['datetime'] == dt, 'IV_Skew'] = iv_skew
                    result_df.loc[result_df['datetime'] == dt, 'ATM_Straddle_Premium'] = atm_straddle_premium
            
            # Calculate IV skew percentile and ATM straddle premium percentile
            if 'IV_Skew' in result_df.columns and not result_df['IV_Skew'].isna().all():
                # Calculate rolling percentile for IV skew
                result_df['IV_Skew_Percentile'] = result_df['IV_Skew'].rolling(window=self.percentile_window, min_periods=1).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
                )
            
            if 'ATM_Straddle_Premium' in result_df.columns and not result_df['ATM_Straddle_Premium'].isna().all():
                # Calculate rolling percentile for ATM straddle premium
                result_df['ATM_Straddle_Premium_Percentile'] = result_df['ATM_Straddle_Premium'].rolling(window=self.percentile_window, min_periods=1).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
                )
            
            # Calculate IV skew regime
            if 'IV_Skew_Percentile' in result_df.columns:
                result_df['IV_Skew_Regime'] = pd.cut(
                    result_df['IV_Skew_Percentile'],
                    bins=[-float('inf'), 20, 40, 60, 80, float('inf')],
                    labels=['Very_Low', 'Low', 'Neutral', 'High', 'Very_High']
                )
            
            # Calculate ATM straddle premium regime
            if 'ATM_Straddle_Premium_Percentile' in result_df.columns:
                result_df['ATM_Straddle_Premium_Regime'] = pd.cut(
                    result_df['ATM_Straddle_Premium_Percentile'],
                    bins=[-float('inf'), 20, 40, 60, 80, float('inf')],
                    labels=['Very_Low', 'Low', 'Neutral', 'High', 'Very_High']
                )
            
            # Calculate combined IV regime
            if 'IV_Skew_Regime' in result_df.columns and 'ATM_Straddle_Premium_Regime' in result_df.columns:
                # Create a mapping for regime values
                regime_values = {
                    'Very_Low': -2,
                    'Low': -1,
                    'Neutral': 0,
                    'High': 1,
                    'Very_High': 2
                }
                
                # Convert regimes to numeric values
                result_df['IV_Skew_Regime_Value'] = result_df['IV_Skew_Regime'].map(regime_values)
                result_df['ATM_Straddle_Premium_Regime_Value'] = result_df['ATM_Straddle_Premium_Regime'].map(regime_values)
                
                # Calculate combined regime value
                result_df['Combined_IV_Regime_Value'] = result_df['IV_Skew_Regime_Value'] + result_df['ATM_Straddle_Premium_Regime_Value']
                
                # Map combined value to regime
                result_df['Combined_IV_Regime'] = pd.cut(
                    result_df['Combined_IV_Regime_Value'],
                    bins=[-float('inf'), -3, -1, 1, 3, float('inf')],
                    labels=['Strong_Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong_Bullish']
                )
            
            logger.info("Calculated IV skew and ATM straddle indicators")
            
        except Exception as e:
            logger.error(f"Error calculating IV skew and ATM straddle indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def visualize_iv_indicators(self, df):
        """
        Visualize IV skew and ATM straddle indicators
        
        Args:
            df (pd.DataFrame): Dataframe with IV indicators
        """
        logger.info("Visualizing IV skew and ATM straddle indicators")
        
        # Check if required columns exist
        iv_columns = ['ATM_IV', 'OTM_Call_IV', 'OTM_Put_IV', 'IV_Skew']
        premium_columns = ['ATM_Straddle_Premium']
        percentile_columns = ['IV_Skew_Percentile', 'ATM_Straddle_Premium_Percentile']
        regime_columns = ['IV_Skew_Regime', 'ATM_Straddle_Premium_Regime', 'Combined_IV_Regime']
        
        # Check which visualizations we can create
        can_visualize_iv = all(col in df.columns for col in iv_columns)
        can_visualize_premium = all(col in df.columns for col in premium_columns)
        can_visualize_percentiles = all(col in df.columns for col in percentile_columns)
        can_visualize_regimes = all(col in df.columns for col in regime_columns)
        
        if not (can_visualize_iv or can_visualize_premium or can_visualize_percentiles or can_visualize_regimes):
            logger.error("No IV indicators available for visualization")
            return
        
        try:
            # Visualize IV values
            if can_visualize_iv:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['ATM_IV'], label='ATM IV', linewidth=2)
                plt.plot(df['OTM_Call_IV'], label='OTM Call IV', linewidth=2)
                plt.plot(df['OTM_Put_IV'], label='OTM Put IV', linewidth=2)
                
                plt.title('Implied Volatility Values')
                plt.xlabel('Time')
                plt.ylabel('IV')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                iv_plot_path = os.path.join(self.output_dir, 'iv_values.png')
                plt.savefig(iv_plot_path)
                logger.info(f"Saved IV values plot to {iv_plot_path}")
                
                plt.close()
                
                # Visualize IV skew
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['IV_Skew'], label='IV Skew', linewidth=2)
                
                plt.title('IV Skew (OTM Put IV - OTM Call IV)')
                plt.xlabel('Time')
                plt.ylabel('IV Skew')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                skew_plot_path = os.path.join(self.output_dir, 'iv_skew.png')
                plt.savefig(skew_plot_path)
                logger.info(f"Saved IV skew plot to {skew_plot_path}")
                
                plt.close()
            
            # Visualize ATM straddle premium
            if can_visualize_premium:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['ATM_Straddle_Premium'], label='ATM Straddle Premium', linewidth=2)
                
                plt.title('ATM Straddle Premium')
                plt.xlabel('Time')
                plt.ylabel('Premium')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                premium_plot_path = os.path.join(self.output_dir, 'atm_straddle_premium.png')
                plt.savefig(premium_plot_path)
                logger.info(f"Saved ATM straddle premium plot to {premium_plot_path}")
                
                plt.close()
            
            # Visualize percentiles
            if can_visualize_percentiles:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['IV_Skew_Percentile'], label='IV Skew Percentile', linewidth=2)
                plt.plot(df['ATM_Straddle_Premium_Percentile'], label='ATM Straddle Premium Percentile', linewidth=2)
                
                plt.title('IV Indicators Percentiles')
                plt.xlabel('Time')
                plt.ylabel('Percentile')
                plt.axhline(y=20, color='r', linestyle='--', alpha=0.3)
                plt.axhline(y=80, color='r', linestyle='--', alpha=0.3)
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                percentile_plot_path = os.path.join(self.output_dir, 'iv_percentiles.png')
                plt.savefig(percentile_plot_path)
                logger.info(f"Saved IV percentiles plot to {percentile_plot_path}")
                
                plt.close()
            
            # Visualize regimes
            if can_visualize_regimes:
                # Create a figure for regime distribution
                plt.figure(figsize=(15, 10))
                
                # Create subplots
                plt.subplot(3, 1, 1)
                iv_skew_counts = df['IV_Skew_Regime'].value_counts().sort_index()
                plt.bar(iv_skew_counts.index, iv_skew_counts.values)
                plt.title('IV Skew Regime Distribution')
                plt.ylabel('Count')
                
                plt.subplot(3, 1, 2)
                premium_counts = df['ATM_Straddle_Premium_Regime'].value_counts().sort_index()
                plt.bar(premium_counts.index, premium_counts.values)
                plt.title('ATM Straddle Premium Regime Distribution')
                plt.ylabel('Count')
                
                plt.subplot(3, 1, 3)
                combined_counts = df['Combined_IV_Regime'].value_counts().sort_index()
                plt.bar(combined_counts.index, combined_counts.values)
                plt.title('Combined IV Regime Distribution')
                plt.ylabel('Count')
                
                plt.tight_layout()
                
                # Save plot
                regime_plot_path = os.path.join(self.output_dir, 'iv_regimes_distribution.png')
                plt.savefig(regime_plot_path)
                logger.info(f"Saved IV regimes distribution plot to {regime_plot_path}")
                
                plt.close()
                
                # Create a figure for combined regime over time
                if 'Underlying_Price' in df.columns and 'Combined_IV_Regime' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Create a colormap for regimes
                    regime_colors = {
                        'Strong_Bullish': 'green',
                        'Bullish': 'lightgreen',
                        'Neutral': 'gray',
                        'Bearish': 'lightcoral',
                        'Strong_Bearish': 'red'
                    }
                    
                    # Plot price
                    if 'Underlying_Price' in df.columns:
                        plt.plot(df['Underlying_Price'], label='Price', alpha=0.5, color='blue')
                    
                    # Plot regime as background color
                    x_points = range(len(df))
                    
                    # Plot colored background for each regime
                    for regime, color in regime_colors.items():
                        mask = df['Combined_IV_Regime'] == regime
                        if mask.any():
                            if 'Underlying_Price' in df.columns:
                                plt.fill_between(x_points, 0, df['Underlying_Price'].max(), where=mask, color=color, alpha=0.2, label=regime)
                            else:
                                plt.fill_between(x_points, 0, 1, where=mask, color=color, alpha=0.2, label=regime)
                    
                    plt.title('Combined IV Regime Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    regime_time_plot_path = os.path.join(self.output_dir, 'combined_iv_regime_time.png')
                    plt.savefig(regime_time_plot_path)
                    logger.info(f"Saved combined IV regime over time plot to {regime_time_plot_path}")
                    
                    plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing IV indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_iv_indicators(self):
        """
        Test IV skew and ATM straddle indicators
        
        Returns:
            pd.DataFrame: Dataframe with IV indicators
        """
        logger.info("Testing IV skew and ATM straddle indicators")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None
        
        # Calculate IV indicators
        result_df = self.calculate_iv_skew_indicators(df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "iv_indicators_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved IV indicators results to {output_path}")
        
        # Visualize results
        self.visualize_iv_indicators(result_df)
        
        # Log summary statistics
        if 'Combined_IV_Regime' in result_df.columns:
            regime_counts = result_df['Combined_IV_Regime'].value_counts()
            logger.info(f"Combined IV Regime distribution: {regime_counts.to_dict()}")
        
        logger.info("IV skew and ATM straddle indicators testing completed")
        
        return result_df

def main():
    """
    Main function to run the IV skew and ATM straddle indicators testing
    """
    logger.info("Starting IV skew and ATM straddle indicators testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/iv_skew_atm_straddle',
        'otm_call_delta': 0.25,
        'otm_put_delta': 0.25,
        'lookback_period': 20,
        'percentile_window': 252
    }
    
    # Create IV indicator tester
    tester = IVSkewATMStraddleTester(config)
    
    # Test IV indicators
    result_df = tester.test_iv_indicators()
    
    logger.info("IV skew and ATM straddle indicators testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
