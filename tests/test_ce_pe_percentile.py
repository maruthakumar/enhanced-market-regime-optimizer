"""
Script to test CE/PE percentile calculations for market regime testing
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

class CEPEPercentileTester:
    """
    Class to test CE/PE percentile calculations for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the CE/PE percentile tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/ce_pe_percentile')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # CE/PE percentile parameters
        self.percentile_window = self.config.get('percentile_window', 252)  # Approximately 1 year of trading days
        self.lookback_period = self.config.get('lookback_period', 20)
        self.atm_range = self.config.get('atm_range', 0.05)  # 5% range around ATM
        
        logger.info(f"Initialized CEPEPercentileTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"CE/PE percentile parameters: percentile_window={self.percentile_window}, lookback_period={self.lookback_period}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for CE/PE percentile testing")
        
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
    
    def calculate_ce_pe_percentiles(self, df):
        """
        Calculate CE/PE percentiles
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with CE/PE percentiles
        """
        logger.info("Calculating CE/PE percentiles")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_columns = ['Strike', 'Option_Type', 'Underlying_Price', 'Open_Interest']
        
        # If Underlying_Price doesn't exist but Price does, use Price
        if 'Underlying_Price' not in result_df.columns and 'Price' in result_df.columns:
            logger.info("Using 'Price' column as 'Underlying_Price'")
            result_df['Underlying_Price'] = result_df['Price']
        
        # Check if required columns exist after adjustments
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns after adjustments: {missing_columns}")
            
            # If Open_Interest is missing, try to create synthetic data
            if 'Open_Interest' in missing_columns:
                logger.warning("Open_Interest column missing, creating synthetic OI data")
                
                # Create synthetic OI based on strike distance from ATM
                if 'Underlying_Price' in result_df.columns and 'Strike' in result_df.columns:
                    # Calculate moneyness
                    result_df['Moneyness'] = result_df['Strike'] / result_df['Underlying_Price']
                    
                    # Create synthetic OI with a bell curve distribution
                    # ATM options have higher OI, far OTM/ITM options have lower OI
                    result_df['Open_Interest'] = 1000 * np.exp(-5 * (result_df['Moneyness'] - 1)**2)
                    
                    # Add some randomness
                    np.random.seed(42)  # For reproducibility
                    result_df['Open_Interest'] = result_df['Open_Interest'] * (1 + np.random.normal(0, 0.2, len(result_df)))
                    
                    # Ensure OI is positive and integer
                    result_df['Open_Interest'] = result_df['Open_Interest'].clip(lower=1).astype(int)
                    
                    logger.info("Created synthetic Open_Interest data based on moneyness")
                    missing_columns.remove('Open_Interest')
        
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
            # Process data by datetime to calculate CE/PE percentiles
            if 'datetime' in result_df.columns:
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize result columns
                result_df['CE_OI'] = np.nan
                result_df['PE_OI'] = np.nan
                result_df['CE_PE_Ratio'] = np.nan
                result_df['CE_PE_Ratio_Percentile'] = np.nan
                result_df['ATM_CE_OI'] = np.nan
                result_df['ATM_PE_OI'] = np.nan
                result_df['ATM_CE_PE_Ratio'] = np.nan
                result_df['ATM_CE_PE_Ratio_Percentile'] = np.nan
                
                # Process each datetime group
                for dt, group in datetime_groups:
                    # Calculate total CE and PE OI
                    ce_group = group[group['Option_Type'].str.lower() == 'call']
                    pe_group = group[group['Option_Type'].str.lower() == 'put']
                    
                    ce_oi = ce_group['Open_Interest'].sum() if len(ce_group) > 0 else 0
                    pe_oi = pe_group['Open_Interest'].sum() if len(pe_group) > 0 else 0
                    
                    # Calculate CE/PE ratio
                    ce_pe_ratio = ce_oi / pe_oi if pe_oi > 0 else np.nan
                    
                    # Find ATM strikes
                    underlying_price = group['Underlying_Price'].iloc[0]
                    atm_lower = underlying_price * (1 - self.atm_range)
                    atm_upper = underlying_price * (1 + self.atm_range)
                    
                    # Get ATM options
                    atm_ce_group = ce_group[(ce_group['Strike'] >= atm_lower) & (ce_group['Strike'] <= atm_upper)]
                    atm_pe_group = pe_group[(pe_group['Strike'] >= atm_lower) & (pe_group['Strike'] <= atm_upper)]
                    
                    atm_ce_oi = atm_ce_group['Open_Interest'].sum() if len(atm_ce_group) > 0 else 0
                    atm_pe_oi = atm_pe_group['Open_Interest'].sum() if len(atm_pe_group) > 0 else 0
                    
                    # Calculate ATM CE/PE ratio
                    atm_ce_pe_ratio = atm_ce_oi / atm_pe_oi if atm_pe_oi > 0 else np.nan
                    
                    # Update result dataframe
                    result_df.loc[result_df['datetime'] == dt, 'CE_OI'] = ce_oi
                    result_df.loc[result_df['datetime'] == dt, 'PE_OI'] = pe_oi
                    result_df.loc[result_df['datetime'] == dt, 'CE_PE_Ratio'] = ce_pe_ratio
                    result_df.loc[result_df['datetime'] == dt, 'ATM_CE_OI'] = atm_ce_oi
                    result_df.loc[result_df['datetime'] == dt, 'ATM_PE_OI'] = atm_pe_oi
                    result_df.loc[result_df['datetime'] == dt, 'ATM_CE_PE_Ratio'] = atm_ce_pe_ratio
            
            # Calculate CE/PE ratio percentiles
            if 'CE_PE_Ratio' in result_df.columns and not result_df['CE_PE_Ratio'].isna().all():
                # Calculate rolling percentile for CE/PE ratio
                result_df['CE_PE_Ratio_Percentile'] = result_df['CE_PE_Ratio'].rolling(window=self.percentile_window, min_periods=1).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
                )
            
            if 'ATM_CE_PE_Ratio' in result_df.columns and not result_df['ATM_CE_PE_Ratio'].isna().all():
                # Calculate rolling percentile for ATM CE/PE ratio
                result_df['ATM_CE_PE_Ratio_Percentile'] = result_df['ATM_CE_PE_Ratio'].rolling(window=self.percentile_window, min_periods=1).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan
                )
            
            # Calculate CE/PE ratio regime
            if 'CE_PE_Ratio_Percentile' in result_df.columns:
                result_df['CE_PE_Ratio_Regime'] = pd.cut(
                    result_df['CE_PE_Ratio_Percentile'],
                    bins=[-float('inf'), 20, 40, 60, 80, float('inf')],
                    labels=['Very_Low', 'Low', 'Neutral', 'High', 'Very_High']
                )
            
            # Calculate ATM CE/PE ratio regime
            if 'ATM_CE_PE_Ratio_Percentile' in result_df.columns:
                result_df['ATM_CE_PE_Ratio_Regime'] = pd.cut(
                    result_df['ATM_CE_PE_Ratio_Percentile'],
                    bins=[-float('inf'), 20, 40, 60, 80, float('inf')],
                    labels=['Very_Low', 'Low', 'Neutral', 'High', 'Very_High']
                )
            
            # Calculate combined CE/PE regime
            if 'CE_PE_Ratio_Regime' in result_df.columns and 'ATM_CE_PE_Ratio_Regime' in result_df.columns:
                # Create a mapping for regime values
                regime_values = {
                    'Very_Low': -2,
                    'Low': -1,
                    'Neutral': 0,
                    'High': 1,
                    'Very_High': 2
                }
                
                # Convert regimes to numeric values
                result_df['CE_PE_Ratio_Regime_Value'] = result_df['CE_PE_Ratio_Regime'].map(regime_values)
                result_df['ATM_CE_PE_Ratio_Regime_Value'] = result_df['ATM_CE_PE_Ratio_Regime'].map(regime_values)
                
                # Calculate combined regime value
                result_df['Combined_CE_PE_Regime_Value'] = result_df['CE_PE_Ratio_Regime_Value'] + result_df['ATM_CE_PE_Ratio_Regime_Value']
                
                # Map combined value to regime
                # Note: High CE/PE ratio is typically bearish (more call buying relative to puts)
                # Low CE/PE ratio is typically bullish (more put buying relative to calls)
                result_df['Combined_CE_PE_Regime'] = pd.cut(
                    result_df['Combined_CE_PE_Regime_Value'],
                    bins=[-float('inf'), -3, -1, 1, 3, float('inf')],
                    labels=['Strong_Bullish', 'Bullish', 'Neutral', 'Bearish', 'Strong_Bearish']
                )
            
            logger.info("Calculated CE/PE percentiles")
            
        except Exception as e:
            logger.error(f"Error calculating CE/PE percentiles: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def visualize_ce_pe_indicators(self, df):
        """
        Visualize CE/PE indicators
        
        Args:
            df (pd.DataFrame): Dataframe with CE/PE indicators
        """
        logger.info("Visualizing CE/PE indicators")
        
        # Check if required columns exist
        oi_columns = ['CE_OI', 'PE_OI', 'ATM_CE_OI', 'ATM_PE_OI']
        ratio_columns = ['CE_PE_Ratio', 'ATM_CE_PE_Ratio']
        percentile_columns = ['CE_PE_Ratio_Percentile', 'ATM_CE_PE_Ratio_Percentile']
        regime_columns = ['CE_PE_Ratio_Regime', 'ATM_CE_PE_Ratio_Regime', 'Combined_CE_PE_Regime']
        
        # Check which visualizations we can create
        can_visualize_oi = all(col in df.columns for col in oi_columns)
        can_visualize_ratio = all(col in df.columns for col in ratio_columns)
        can_visualize_percentiles = all(col in df.columns for col in percentile_columns)
        can_visualize_regimes = all(col in df.columns for col in regime_columns)
        
        if not (can_visualize_oi or can_visualize_ratio or can_visualize_percentiles or can_visualize_regimes):
            logger.error("No CE/PE indicators available for visualization")
            return
        
        try:
            # Visualize OI values
            if can_visualize_oi:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['CE_OI'], label='CE OI', linewidth=2)
                plt.plot(df['PE_OI'], label='PE OI', linewidth=2)
                
                plt.title('CE and PE Open Interest')
                plt.xlabel('Time')
                plt.ylabel('Open Interest')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                oi_plot_path = os.path.join(self.output_dir, 'ce_pe_oi.png')
                plt.savefig(oi_plot_path)
                logger.info(f"Saved CE/PE OI plot to {oi_plot_path}")
                
                plt.close()
                
                # Visualize ATM OI values
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['ATM_CE_OI'], label='ATM CE OI', linewidth=2)
                plt.plot(df['ATM_PE_OI'], label='ATM PE OI', linewidth=2)
                
                plt.title('ATM CE and PE Open Interest')
                plt.xlabel('Time')
                plt.ylabel('Open Interest')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                atm_oi_plot_path = os.path.join(self.output_dir, 'atm_ce_pe_oi.png')
                plt.savefig(atm_oi_plot_path)
                logger.info(f"Saved ATM CE/PE OI plot to {atm_oi_plot_path}")
                
                plt.close()
            
            # Visualize CE/PE ratios
            if can_visualize_ratio:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['CE_PE_Ratio'], label='CE/PE Ratio', linewidth=2)
                plt.plot(df['ATM_CE_PE_Ratio'], label='ATM CE/PE Ratio', linewidth=2)
                
                plt.title('CE/PE Ratios')
                plt.xlabel('Time')
                plt.ylabel('Ratio')
                plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                ratio_plot_path = os.path.join(self.output_dir, 'ce_pe_ratios.png')
                plt.savefig(ratio_plot_path)
                logger.info(f"Saved CE/PE ratios plot to {ratio_plot_path}")
                
                plt.close()
            
            # Visualize percentiles
            if can_visualize_percentiles:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['CE_PE_Ratio_Percentile'], label='CE/PE Ratio Percentile', linewidth=2)
                plt.plot(df['ATM_CE_PE_Ratio_Percentile'], label='ATM CE/PE Ratio Percentile', linewidth=2)
                
                plt.title('CE/PE Ratio Percentiles')
                plt.xlabel('Time')
                plt.ylabel('Percentile')
                plt.axhline(y=20, color='r', linestyle='--', alpha=0.3)
                plt.axhline(y=80, color='r', linestyle='--', alpha=0.3)
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                percentile_plot_path = os.path.join(self.output_dir, 'ce_pe_percentiles.png')
                plt.savefig(percentile_plot_path)
                logger.info(f"Saved CE/PE percentiles plot to {percentile_plot_path}")
                
                plt.close()
            
            # Visualize regimes
            if can_visualize_regimes:
                # Create a figure for regime distribution
                plt.figure(figsize=(15, 10))
                
                # Create subplots
                plt.subplot(3, 1, 1)
                ce_pe_counts = df['CE_PE_Ratio_Regime'].value_counts().sort_index()
                plt.bar(ce_pe_counts.index, ce_pe_counts.values)
                plt.title('CE/PE Ratio Regime Distribution')
                plt.ylabel('Count')
                
                plt.subplot(3, 1, 2)
                atm_ce_pe_counts = df['ATM_CE_PE_Ratio_Regime'].value_counts().sort_index()
                plt.bar(atm_ce_pe_counts.index, atm_ce_pe_counts.values)
                plt.title('ATM CE/PE Ratio Regime Distribution')
                plt.ylabel('Count')
                
                plt.subplot(3, 1, 3)
                combined_counts = df['Combined_CE_PE_Regime'].value_counts().sort_index()
                plt.bar(combined_counts.index, combined_counts.values)
                plt.title('Combined CE/PE Regime Distribution')
                plt.ylabel('Count')
                
                plt.tight_layout()
                
                # Save plot
                regime_plot_path = os.path.join(self.output_dir, 'ce_pe_regimes_distribution.png')
                plt.savefig(regime_plot_path)
                logger.info(f"Saved CE/PE regimes distribution plot to {regime_plot_path}")
                
                plt.close()
                
                # Create a figure for combined regime over time
                if 'Underlying_Price' in df.columns and 'Combined_CE_PE_Regime' in df.columns:
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
                        mask = df['Combined_CE_PE_Regime'] == regime
                        if mask.any():
                            if 'Underlying_Price' in df.columns:
                                plt.fill_between(x_points, 0, df['Underlying_Price'].max(), where=mask, color=color, alpha=0.2, label=regime)
                            else:
                                plt.fill_between(x_points, 0, 1, where=mask, color=color, alpha=0.2, label=regime)
                    
                    plt.title('Combined CE/PE Regime Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    regime_time_plot_path = os.path.join(self.output_dir, 'combined_ce_pe_regime_time.png')
                    plt.savefig(regime_time_plot_path)
                    logger.info(f"Saved combined CE/PE regime over time plot to {regime_time_plot_path}")
                    
                    plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing CE/PE indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_ce_pe_percentiles(self):
        """
        Test CE/PE percentile calculations
        
        Returns:
            pd.DataFrame: Dataframe with CE/PE percentiles
        """
        logger.info("Testing CE/PE percentile calculations")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None
        
        # Calculate CE/PE percentiles
        result_df = self.calculate_ce_pe_percentiles(df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "ce_pe_percentile_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved CE/PE percentile results to {output_path}")
        
        # Visualize results
        self.visualize_ce_pe_indicators(result_df)
        
        # Log summary statistics
        if 'Combined_CE_PE_Regime' in result_df.columns:
            regime_counts = result_df['Combined_CE_PE_Regime'].value_counts()
            logger.info(f"Combined CE/PE Regime distribution: {regime_counts.to_dict()}")
        
        logger.info("CE/PE percentile calculations testing completed")
        
        return result_df

def main():
    """
    Main function to run the CE/PE percentile calculations testing
    """
    logger.info("Starting CE/PE percentile calculations testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/ce_pe_percentile',
        'percentile_window': 252,
        'lookback_period': 20,
        'atm_range': 0.05
    }
    
    # Create CE/PE percentile tester
    tester = CEPEPercentileTester(config)
    
    # Test CE/PE percentiles
    result_df = tester.test_ce_pe_percentiles()
    
    logger.info("CE/PE percentile calculations testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
