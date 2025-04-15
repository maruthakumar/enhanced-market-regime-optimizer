"""
Script to test EMA indicators for market regime testing
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

class EMAIndicatorTester:
    """
    Class to test EMA indicators for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the EMA indicator tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/ema_indicators')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # EMA parameters
        self.short_ema_period = self.config.get('short_ema_period', 5)
        self.medium_ema_period = self.config.get('medium_ema_period', 20)
        self.long_ema_period = self.config.get('long_ema_period', 50)
        
        logger.info(f"Initialized EMAIndicatorTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"EMA periods: short={self.short_ema_period}, medium={self.medium_ema_period}, long={self.long_ema_period}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for EMA indicator testing")
        
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
    
    def calculate_ema_indicators(self, df):
        """
        Calculate EMA indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with EMA indicators
        """
        logger.info("Calculating EMA indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        if 'Underlying_Price' not in result_df.columns and 'Price' in result_df.columns:
            logger.info("Using 'Price' column as 'Underlying_Price'")
            result_df['Underlying_Price'] = result_df['Price']
        
        if 'Underlying_Price' not in result_df.columns:
            logger.error("Required column 'Underlying_Price' not found")
            return result_df
        
        # Ensure datetime is in datetime format
        if 'datetime' in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df['datetime']):
            try:
                result_df['datetime'] = pd.to_datetime(result_df['datetime'])
                logger.info("Converted datetime to datetime format")
            except Exception as e:
                logger.warning(f"Failed to convert datetime to datetime format: {str(e)}")
        
        # Sort by datetime
        if 'datetime' in result_df.columns:
            result_df.sort_values('datetime', inplace=True)
            logger.info("Sorted data by datetime")
        
        # Calculate EMAs
        try:
            # Calculate short EMA
            result_df[f'EMA_{self.short_ema_period}'] = result_df['Underlying_Price'].ewm(span=self.short_ema_period, adjust=False).mean()
            logger.info(f"Calculated EMA_{self.short_ema_period}")
            
            # Calculate medium EMA
            result_df[f'EMA_{self.medium_ema_period}'] = result_df['Underlying_Price'].ewm(span=self.medium_ema_period, adjust=False).mean()
            logger.info(f"Calculated EMA_{self.medium_ema_period}")
            
            # Calculate long EMA
            result_df[f'EMA_{self.long_ema_period}'] = result_df['Underlying_Price'].ewm(span=self.long_ema_period, adjust=False).mean()
            logger.info(f"Calculated EMA_{self.long_ema_period}")
            
            # Calculate EMA crossovers
            result_df['EMA_Short_Medium_Crossover'] = np.where(
                result_df[f'EMA_{self.short_ema_period}'] > result_df[f'EMA_{self.medium_ema_period}'], 1,
                np.where(result_df[f'EMA_{self.short_ema_period}'] < result_df[f'EMA_{self.medium_ema_period}'], -1, 0)
            )
            
            result_df['EMA_Medium_Long_Crossover'] = np.where(
                result_df[f'EMA_{self.medium_ema_period}'] > result_df[f'EMA_{self.long_ema_period}'], 1,
                np.where(result_df[f'EMA_{self.medium_ema_period}'] < result_df[f'EMA_{self.long_ema_period}'], -1, 0)
            )
            
            # Calculate EMA trend
            result_df['EMA_Trend'] = np.where(
                (result_df['EMA_Short_Medium_Crossover'] == 1) & (result_df['EMA_Medium_Long_Crossover'] == 1), 'Strong_Bullish',
                np.where((result_df['EMA_Short_Medium_Crossover'] == 1) & (result_df['EMA_Medium_Long_Crossover'] == -1), 'Weak_Bullish',
                np.where((result_df['EMA_Short_Medium_Crossover'] == -1) & (result_df['EMA_Medium_Long_Crossover'] == 1), 'Weak_Bearish',
                np.where((result_df['EMA_Short_Medium_Crossover'] == -1) & (result_df['EMA_Medium_Long_Crossover'] == -1), 'Strong_Bearish',
                'Neutral')))
            )
            
            logger.info("Calculated EMA crossovers and trend")
            
        except Exception as e:
            logger.error(f"Error calculating EMA indicators: {str(e)}")
        
        return result_df
    
    def visualize_ema_indicators(self, df):
        """
        Visualize EMA indicators
        
        Args:
            df (pd.DataFrame): Dataframe with EMA indicators
        """
        logger.info("Visualizing EMA indicators")
        
        # Check if required columns exist
        required_columns = [f'EMA_{self.short_ema_period}', f'EMA_{self.medium_ema_period}', f'EMA_{self.long_ema_period}', 'Underlying_Price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for visualization: {missing_columns}")
            return
        
        try:
            # Create a figure for EMA lines
            plt.figure(figsize=(12, 6))
            
            # Plot price and EMAs
            plt.plot(df['Underlying_Price'], label='Price', alpha=0.5)
            plt.plot(df[f'EMA_{self.short_ema_period}'], label=f'EMA {self.short_ema_period}', linewidth=2)
            plt.plot(df[f'EMA_{self.medium_ema_period}'], label=f'EMA {self.medium_ema_period}', linewidth=2)
            plt.plot(df[f'EMA_{self.long_ema_period}'], label=f'EMA {self.long_ema_period}', linewidth=2)
            
            plt.title('EMA Indicators')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save plot
            ema_plot_path = os.path.join(self.output_dir, 'ema_indicators.png')
            plt.savefig(ema_plot_path)
            logger.info(f"Saved EMA plot to {ema_plot_path}")
            
            plt.close()
            
            # Create a figure for EMA trend
            plt.figure(figsize=(12, 6))
            
            # Create a colormap for EMA trend
            trend_colors = {
                'Strong_Bullish': 'green',
                'Weak_Bullish': 'lightgreen',
                'Neutral': 'gray',
                'Weak_Bearish': 'lightcoral',
                'Strong_Bearish': 'red'
            }
            
            # Plot price
            plt.plot(df['Underlying_Price'], label='Price', alpha=0.5, color='blue')
            
            # Plot EMA trend as background color
            if 'EMA_Trend' in df.columns:
                # Get unique dates or x-axis points
                x_points = range(len(df))
                
                # Plot colored background for each trend
                for trend, color in trend_colors.items():
                    mask = df['EMA_Trend'] == trend
                    if mask.any():
                        plt.fill_between(x_points, 0, df['Underlying_Price'].max(), where=mask, color=color, alpha=0.2, label=trend)
            
            plt.title('EMA Trend')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save plot
            trend_plot_path = os.path.join(self.output_dir, 'ema_trend.png')
            plt.savefig(trend_plot_path)
            logger.info(f"Saved EMA trend plot to {trend_plot_path}")
            
            plt.close()
            
            # Create a figure for EMA trend distribution
            if 'EMA_Trend' in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Count occurrences of each trend
                trend_counts = df['EMA_Trend'].value_counts()
                
                # Plot trend distribution
                bars = plt.bar(trend_counts.index, trend_counts.values, color=[trend_colors.get(trend, 'gray') for trend in trend_counts.index])
                
                plt.title('EMA Trend Distribution')
                plt.xlabel('Trend')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save plot
                dist_plot_path = os.path.join(self.output_dir, 'ema_trend_distribution.png')
                plt.savefig(dist_plot_path)
                logger.info(f"Saved EMA trend distribution plot to {dist_plot_path}")
                
                plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing EMA indicators: {str(e)}")
    
    def test_ema_indicators(self):
        """
        Test EMA indicators
        
        Returns:
            pd.DataFrame: Dataframe with EMA indicators
        """
        logger.info("Testing EMA indicators")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None
        
        # Calculate EMA indicators
        result_df = self.calculate_ema_indicators(df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "ema_indicators_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved EMA indicators results to {output_path}")
        
        # Visualize results
        self.visualize_ema_indicators(result_df)
        
        # Log summary statistics
        if 'EMA_Trend' in result_df.columns:
            trend_counts = result_df['EMA_Trend'].value_counts()
            logger.info(f"EMA Trend distribution: {trend_counts.to_dict()}")
        
        logger.info("EMA indicators testing completed")
        
        return result_df

def main():
    """
    Main function to run the EMA indicators testing
    """
    logger.info("Starting EMA indicators testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/ema_indicators',
        'short_ema_period': 5,
        'medium_ema_period': 20,
        'long_ema_period': 50
    }
    
    # Create EMA indicator tester
    tester = EMAIndicatorTester(config)
    
    # Test EMA indicators
    result_df = tester.test_ema_indicators()
    
    logger.info("EMA indicators testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
