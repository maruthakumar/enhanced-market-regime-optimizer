"""
Script to test VWAP indicators for market regime testing
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

class VWAPIndicatorTester:
    """
    Class to test VWAP indicators for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the VWAP indicator tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/vwap_indicators')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # VWAP parameters
        self.short_vwap_period = self.config.get('short_vwap_period', 15)  # 15 minutes
        self.medium_vwap_period = self.config.get('medium_vwap_period', 60)  # 1 hour
        self.long_vwap_period = self.config.get('long_vwap_period', 240)  # 4 hours
        
        logger.info(f"Initialized VWAPIndicatorTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"VWAP periods: short={self.short_vwap_period}, medium={self.medium_vwap_period}, long={self.long_vwap_period}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for VWAP indicator testing")
        
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
    
    def calculate_vwap_indicators(self, df):
        """
        Calculate VWAP indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with VWAP indicators
        """
        logger.info("Calculating VWAP indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_columns = ['Underlying_Price', 'Volume']
        
        # If Underlying_Price doesn't exist but Price does, use Price
        if 'Underlying_Price' not in result_df.columns and 'Price' in result_df.columns:
            logger.info("Using 'Price' column as 'Underlying_Price'")
            result_df['Underlying_Price'] = result_df['Price']
        
        # If Volume doesn't exist, create a synthetic one
        if 'Volume' not in result_df.columns:
            logger.warning("Volume column not found, creating synthetic volume data")
            # Create random volume data with some correlation to price changes
            np.random.seed(42)  # For reproducibility
            result_df['Volume'] = np.random.randint(100, 1000, size=len(result_df))
            
            # Add some correlation with price changes if possible
            if 'Underlying_Price' in result_df.columns:
                price_diff = result_df['Underlying_Price'].diff().fillna(0)
                # Scale price differences to reasonable volume changes
                volume_factor = price_diff.abs() * 10
                result_df['Volume'] = result_df['Volume'] + volume_factor.astype(int)
                result_df['Volume'] = result_df['Volume'].clip(lower=100)  # Ensure minimum volume
        
        # Check if required columns exist after adjustments
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns after adjustments: {missing_columns}")
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
            # Calculate typical price (TP = (High + Low + Close) / 3)
            # If High and Low are not available, use Close/Price as typical price
            if 'High' in result_df.columns and 'Low' in result_df.columns:
                result_df['TypicalPrice'] = (result_df['High'] + result_df['Low'] + result_df['Underlying_Price']) / 3
                logger.info("Calculated typical price using High, Low, and Underlying_Price")
            else:
                result_df['TypicalPrice'] = result_df['Underlying_Price']
                logger.info("Using Underlying_Price as typical price")
            
            # Calculate VWAP components
            result_df['TP_Volume'] = result_df['TypicalPrice'] * result_df['Volume']
            result_df['CumVolume'] = result_df['Volume'].cumsum()
            result_df['CumTP_Volume'] = result_df['TP_Volume'].cumsum()
            
            # Calculate daily VWAP
            result_df['VWAP_Daily'] = result_df['CumTP_Volume'] / result_df['CumVolume']
            
            # Calculate short, medium, and long period VWAPs
            # For rolling windows, we need to ensure datetime is properly set
            if 'datetime' in result_df.columns:
                # Group by date to reset cumulative calculations each day
                if 'Date' in result_df.columns:
                    result_df['Date'] = pd.to_datetime(result_df['Date']).dt.date
                    date_groups = result_df.groupby('Date')
                    
                    # Initialize VWAP columns
                    result_df[f'VWAP_{self.short_vwap_period}'] = np.nan
                    result_df[f'VWAP_{self.medium_vwap_period}'] = np.nan
                    result_df[f'VWAP_{self.long_vwap_period}'] = np.nan
                    
                    # Calculate VWAPs for each date group
                    for date, group in date_groups:
                        # Get group indices
                        idx = group.index
                        
                        # Calculate rolling VWAPs
                        tp_vol_rolling_short = group['TP_Volume'].rolling(window=self.short_vwap_period, min_periods=1).sum()
                        vol_rolling_short = group['Volume'].rolling(window=self.short_vwap_period, min_periods=1).sum()
                        
                        tp_vol_rolling_medium = group['TP_Volume'].rolling(window=self.medium_vwap_period, min_periods=1).sum()
                        vol_rolling_medium = group['Volume'].rolling(window=self.medium_vwap_period, min_periods=1).sum()
                        
                        tp_vol_rolling_long = group['TP_Volume'].rolling(window=self.long_vwap_period, min_periods=1).sum()
                        vol_rolling_long = group['Volume'].rolling(window=self.long_vwap_period, min_periods=1).sum()
                        
                        # Calculate VWAPs
                        result_df.loc[idx, f'VWAP_{self.short_vwap_period}'] = tp_vol_rolling_short / vol_rolling_short
                        result_df.loc[idx, f'VWAP_{self.medium_vwap_period}'] = tp_vol_rolling_medium / vol_rolling_medium
                        result_df.loc[idx, f'VWAP_{self.long_vwap_period}'] = tp_vol_rolling_long / vol_rolling_long
                else:
                    # If Date column doesn't exist, calculate rolling VWAPs without daily reset
                    tp_vol_rolling_short = result_df['TP_Volume'].rolling(window=self.short_vwap_period, min_periods=1).sum()
                    vol_rolling_short = result_df['Volume'].rolling(window=self.short_vwap_period, min_periods=1).sum()
                    
                    tp_vol_rolling_medium = result_df['TP_Volume'].rolling(window=self.medium_vwap_period, min_periods=1).sum()
                    vol_rolling_medium = result_df['Volume'].rolling(window=self.medium_vwap_period, min_periods=1).sum()
                    
                    tp_vol_rolling_long = result_df['TP_Volume'].rolling(window=self.long_vwap_period, min_periods=1).sum()
                    vol_rolling_long = result_df['Volume'].rolling(window=self.long_vwap_period, min_periods=1).sum()
                    
                    # Calculate VWAPs
                    result_df[f'VWAP_{self.short_vwap_period}'] = tp_vol_rolling_short / vol_rolling_short
                    result_df[f'VWAP_{self.medium_vwap_period}'] = tp_vol_rolling_medium / vol_rolling_medium
                    result_df[f'VWAP_{self.long_vwap_period}'] = tp_vol_rolling_long / vol_rolling_long
            
            # Calculate VWAP crossovers
            result_df['VWAP_Short_Medium_Crossover'] = np.where(
                result_df[f'VWAP_{self.short_vwap_period}'] > result_df[f'VWAP_{self.medium_vwap_period}'], 1,
                np.where(result_df[f'VWAP_{self.short_vwap_period}'] < result_df[f'VWAP_{self.medium_vwap_period}'], -1, 0)
            )
            
            result_df['VWAP_Medium_Long_Crossover'] = np.where(
                result_df[f'VWAP_{self.medium_vwap_period}'] > result_df[f'VWAP_{self.long_vwap_period}'], 1,
                np.where(result_df[f'VWAP_{self.medium_vwap_period}'] < result_df[f'VWAP_{self.long_vwap_period}'], -1, 0)
            )
            
            # Calculate price vs VWAP position
            result_df['Price_vs_VWAP'] = np.where(
                result_df['Underlying_Price'] > result_df['VWAP_Daily'], 1,
                np.where(result_df['Underlying_Price'] < result_df['VWAP_Daily'], -1, 0)
            )
            
            # Calculate VWAP trend
            result_df['VWAP_Trend'] = np.where(
                (result_df['VWAP_Short_Medium_Crossover'] == 1) & (result_df['VWAP_Medium_Long_Crossover'] == 1) & (result_df['Price_vs_VWAP'] == 1), 'Strong_Bullish',
                np.where((result_df['VWAP_Short_Medium_Crossover'] == 1) & (result_df['Price_vs_VWAP'] == 1), 'Bullish',
                np.where((result_df['VWAP_Short_Medium_Crossover'] == -1) & (result_df['VWAP_Medium_Long_Crossover'] == -1) & (result_df['Price_vs_VWAP'] == -1), 'Strong_Bearish',
                np.where((result_df['VWAP_Short_Medium_Crossover'] == -1) & (result_df['Price_vs_VWAP'] == -1), 'Bearish',
                'Neutral')))
            )
            
            logger.info("Calculated VWAP indicators")
            
        except Exception as e:
            logger.error(f"Error calculating VWAP indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def visualize_vwap_indicators(self, df):
        """
        Visualize VWAP indicators
        
        Args:
            df (pd.DataFrame): Dataframe with VWAP indicators
        """
        logger.info("Visualizing VWAP indicators")
        
        # Check if required columns exist
        required_columns = [f'VWAP_{self.short_vwap_period}', f'VWAP_{self.medium_vwap_period}', 
                           f'VWAP_{self.long_vwap_period}', 'VWAP_Daily', 'Underlying_Price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for visualization: {missing_columns}")
            return
        
        try:
            # Create a figure for VWAP lines
            plt.figure(figsize=(12, 6))
            
            # Plot price and VWAPs
            plt.plot(df['Underlying_Price'], label='Price', alpha=0.5)
            plt.plot(df['VWAP_Daily'], label='Daily VWAP', linewidth=2)
            plt.plot(df[f'VWAP_{self.short_vwap_period}'], label=f'VWAP {self.short_vwap_period}', linewidth=2)
            plt.plot(df[f'VWAP_{self.medium_vwap_period}'], label=f'VWAP {self.medium_vwap_period}', linewidth=2)
            plt.plot(df[f'VWAP_{self.long_vwap_period}'], label=f'VWAP {self.long_vwap_period}', linewidth=2)
            
            plt.title('VWAP Indicators')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save plot
            vwap_plot_path = os.path.join(self.output_dir, 'vwap_indicators.png')
            plt.savefig(vwap_plot_path)
            logger.info(f"Saved VWAP plot to {vwap_plot_path}")
            
            plt.close()
            
            # Create a figure for VWAP trend
            plt.figure(figsize=(12, 6))
            
            # Create a colormap for VWAP trend
            trend_colors = {
                'Strong_Bullish': 'green',
                'Bullish': 'lightgreen',
                'Neutral': 'gray',
                'Bearish': 'lightcoral',
                'Strong_Bearish': 'red'
            }
            
            # Plot price
            plt.plot(df['Underlying_Price'], label='Price', alpha=0.5, color='blue')
            
            # Plot VWAP trend as background color
            if 'VWAP_Trend' in df.columns:
                # Get unique dates or x-axis points
                x_points = range(len(df))
                
                # Plot colored background for each trend
                for trend, color in trend_colors.items():
                    mask = df['VWAP_Trend'] == trend
                    if mask.any():
                        plt.fill_between(x_points, 0, df['Underlying_Price'].max(), where=mask, color=color, alpha=0.2, label=trend)
            
            plt.title('VWAP Trend')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save plot
            trend_plot_path = os.path.join(self.output_dir, 'vwap_trend.png')
            plt.savefig(trend_plot_path)
            logger.info(f"Saved VWAP trend plot to {trend_plot_path}")
            
            plt.close()
            
            # Create a figure for VWAP trend distribution
            if 'VWAP_Trend' in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Count occurrences of each trend
                trend_counts = df['VWAP_Trend'].value_counts()
                
                # Plot trend distribution
                bars = plt.bar(trend_counts.index, trend_counts.values, color=[trend_colors.get(trend, 'gray') for trend in trend_counts.index])
                
                plt.title('VWAP Trend Distribution')
                plt.xlabel('Trend')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save plot
                dist_plot_path = os.path.join(self.output_dir, 'vwap_trend_distribution.png')
                plt.savefig(dist_plot_path)
                logger.info(f"Saved VWAP trend distribution plot to {dist_plot_path}")
                
                plt.close()
            
            # Create a figure for Price vs VWAP
            if 'Price_vs_VWAP' in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Plot price and daily VWAP
                plt.plot(df['Underlying_Price'], label='Price', alpha=0.7)
                plt.plot(df['VWAP_Daily'], label='Daily VWAP', linewidth=2)
                
                # Color regions based on Price vs VWAP
                x_points = range(len(df))
                
                # Above VWAP (bullish)
                above_mask = df['Price_vs_VWAP'] == 1
                if above_mask.any():
                    plt.fill_between(x_points, df['VWAP_Daily'], df['Underlying_Price'], where=above_mask, color='green', alpha=0.3, label='Above VWAP')
                
                # Below VWAP (bearish)
                below_mask = df['Price_vs_VWAP'] == -1
                if below_mask.any():
                    plt.fill_between(x_points, df['Underlying_Price'], df['VWAP_Daily'], where=below_mask, color='red', alpha=0.3, label='Below VWAP')
                
                plt.title('Price vs VWAP')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                price_vwap_plot_path = os.path.join(self.output_dir, 'price_vs_vwap.png')
                plt.savefig(price_vwap_plot_path)
                logger.info(f"Saved Price vs VWAP plot to {price_vwap_plot_path}")
                
                plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing VWAP indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_vwap_indicators(self):
        """
        Test VWAP indicators
        
        Returns:
            pd.DataFrame: Dataframe with VWAP indicators
        """
        logger.info("Testing VWAP indicators")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None
        
        # Calculate VWAP indicators
        result_df = self.calculate_vwap_indicators(df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "vwap_indicators_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved VWAP indicators results to {output_path}")
        
        # Visualize results
        self.visualize_vwap_indicators(result_df)
        
        # Log summary statistics
        if 'VWAP_Trend' in result_df.columns:
            trend_counts = result_df['VWAP_Trend'].value_counts()
            logger.info(f"VWAP Trend distribution: {trend_counts.to_dict()}")
        
        logger.info("VWAP indicators testing completed")
        
        return result_df

def main():
    """
    Main function to run the VWAP indicators testing
    """
    logger.info("Starting VWAP indicators testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/vwap_indicators',
        'short_vwap_period': 15,
        'medium_vwap_period': 60,
        'long_vwap_period': 240
    }
    
    # Create VWAP indicator tester
    tester = VWAPIndicatorTester(config)
    
    # Test VWAP indicators
    result_df = tester.test_vwap_indicators()
    
    logger.info("VWAP indicators testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
