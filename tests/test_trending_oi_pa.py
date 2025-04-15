"""
Script to test trending OI with price action analysis for market regime testing
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

class TrendingOIPATester:
    """
    Class to test trending OI with price action analysis for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the trending OI with PA tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/trending_oi_pa')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Trending OI with PA parameters
        self.strikes_above_atm = self.config.get('strikes_above_atm', 7)  # 7 strikes above ATM
        self.strikes_below_atm = self.config.get('strikes_below_atm', 7)  # 7 strikes below ATM
        self.lookback_period = self.config.get('lookback_period', 5)  # 5 periods lookback
        self.time_interval = self.config.get('time_interval', '5min')  # 5-minute intervals
        
        # OI change thresholds
        self.oi_increase_threshold = self.config.get('oi_increase_threshold', 0.05)  # 5% increase
        self.oi_decrease_threshold = self.config.get('oi_decrease_threshold', -0.05)  # 5% decrease
        
        # Price action thresholds
        self.price_increase_threshold = self.config.get('price_increase_threshold', 0.01)  # 1% increase
        self.price_decrease_threshold = self.config.get('price_decrease_threshold', -0.01)  # 1% decrease
        
        logger.info(f"Initialized TrendingOIPATester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"Trending OI with PA parameters: strikes_above_atm={self.strikes_above_atm}, strikes_below_atm={self.strikes_below_atm}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for trending OI with PA testing")
        
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
    
    def calculate_trending_oi_pa(self, df):
        """
        Calculate trending OI with price action indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with trending OI with PA indicators
        """
        logger.info("Calculating trending OI with price action indicators")
        
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
        
        # Sort by datetime and strike
        if 'datetime' in result_df.columns:
            result_df.sort_values(['datetime', 'Strike'], inplace=True)
            logger.info("Sorted data by datetime and strike")
        
        try:
            # Process data by datetime to calculate trending OI with PA
            if 'datetime' in result_df.columns:
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize result dataframes for each timestamp
                result_dfs = []
                
                # Process each datetime group
                for dt, group in datetime_groups:
                    # Find ATM strike
                    underlying_price = group['Underlying_Price'].iloc[0]
                    strikes = sorted(group['Strike'].unique())
                    
                    # Find closest strike to underlying price
                    atm_strike = strikes[np.abs(np.array(strikes) - underlying_price).argmin()]
                    
                    # Select strikes around ATM
                    atm_index = strikes.index(atm_strike)
                    start_index = max(0, atm_index - self.strikes_below_atm)
                    end_index = min(len(strikes), atm_index + self.strikes_above_atm + 1)
                    
                    selected_strikes = strikes[start_index:end_index]
                    
                    # Filter data for selected strikes
                    selected_data = group[group['Strike'].isin(selected_strikes)].copy()
                    
                    # Add datetime to selected data
                    selected_data['datetime'] = dt
                    
                    # Append to result dataframes
                    result_dfs.append(selected_data)
                
                # Combine all result dataframes
                if result_dfs:
                    result_df = pd.concat(result_dfs, ignore_index=True)
                    logger.info(f"Combined data for {len(result_dfs)} timestamps")
                else:
                    logger.warning("No data after processing timestamps")
                    return result_df
                
                # Calculate OI changes between consecutive timestamps
                # Group by strike and option type
                strike_option_groups = result_df.groupby(['Strike', 'Option_Type'])
                
                # Initialize OI change columns
                result_df['OI_Change'] = np.nan
                result_df['OI_Change_Pct'] = np.nan
                
                # Calculate OI changes for each strike and option type
                for (strike, option_type), group in strike_option_groups:
                    # Sort by datetime
                    group = group.sort_values('datetime')
                    
                    # Calculate OI change
                    group['OI_Change'] = group['Open_Interest'].diff()
                    
                    # Calculate OI change percentage
                    group['OI_Change_Pct'] = group['Open_Interest'].pct_change() * 100
                    
                    # Update result dataframe
                    result_df.loc[(result_df['Strike'] == strike) & (result_df['Option_Type'] == option_type), 'OI_Change'] = group['OI_Change'].values
                    result_df.loc[(result_df['Strike'] == strike) & (result_df['Option_Type'] == option_type), 'OI_Change_Pct'] = group['OI_Change_Pct'].values
                
                # Calculate price changes between consecutive timestamps
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize price change columns
                result_df['Price_Change'] = np.nan
                result_df['Price_Change_Pct'] = np.nan
                
                # Get unique timestamps
                timestamps = sorted(result_df['datetime'].unique())
                
                # Calculate price changes for each timestamp
                for i in range(1, len(timestamps)):
                    prev_timestamp = timestamps[i-1]
                    curr_timestamp = timestamps[i]
                    
                    prev_price = result_df[result_df['datetime'] == prev_timestamp]['Underlying_Price'].iloc[0]
                    curr_price = result_df[result_df['datetime'] == curr_timestamp]['Underlying_Price'].iloc[0]
                    
                    price_change = curr_price - prev_price
                    price_change_pct = (curr_price / prev_price - 1) * 100 if prev_price > 0 else 0
                    
                    result_df.loc[result_df['datetime'] == curr_timestamp, 'Price_Change'] = price_change
                    result_df.loc[result_df['datetime'] == curr_timestamp, 'Price_Change_Pct'] = price_change_pct
                
                # Calculate OI patterns based on OI change and price change
                # Initialize OI pattern columns
                result_df['OI_Pattern'] = 'Unknown'
                
                # Calculate OI patterns for each row
                for idx, row in result_df.iterrows():
                    if pd.isna(row['OI_Change']) or pd.isna(row['Price_Change']):
                        continue
                    
                    oi_change = row['OI_Change']
                    price_change = row['Price_Change']
                    option_type = row['Option_Type'].lower()
                    
                    # Determine OI pattern based on OI change and price change
                    if option_type == 'call':
                        if oi_change > 0 and price_change > 0:
                            pattern = 'Long_Build_Up'  # OI up, price up
                        elif oi_change < 0 and price_change < 0:
                            pattern = 'Long_Unwinding'  # OI down, price down
                        elif oi_change > 0 and price_change < 0:
                            pattern = 'Short_Build_Up'  # OI up, price down
                        elif oi_change < 0 and price_change > 0:
                            pattern = 'Short_Covering'  # OI down, price up
                        else:
                            pattern = 'Neutral'
                    elif option_type == 'put':
                        if oi_change > 0 and price_change < 0:
                            pattern = 'Long_Build_Up'  # OI up, price down (for puts)
                        elif oi_change < 0 and price_change > 0:
                            pattern = 'Long_Unwinding'  # OI down, price up (for puts)
                        elif oi_change > 0 and price_change > 0:
                            pattern = 'Short_Build_Up'  # OI up, price up (for puts)
                        elif oi_change < 0 and price_change < 0:
                            pattern = 'Short_Covering'  # OI down, price down (for puts)
                        else:
                            pattern = 'Neutral'
                    else:
                        pattern = 'Unknown'
                    
                    result_df.loc[idx, 'OI_Pattern'] = pattern
                
                # Calculate aggregated OI metrics for each timestamp
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize aggregated columns
                aggregated_df = pd.DataFrame()
                
                # Calculate aggregated metrics for each timestamp
                for dt, group in datetime_groups:
                    # Get call and put data
                    call_data = group[group['Option_Type'].str.lower() == 'call']
                    put_data = group[group['Option_Type'].str.lower() == 'put']
                    
                    # Calculate total OI and OI changes
                    total_call_oi = call_data['Open_Interest'].sum()
                    total_put_oi = put_data['Open_Interest'].sum()
                    
                    call_oi_change = call_data['OI_Change'].sum()
                    put_oi_change = put_data['OI_Change'].sum()
                    
                    # Calculate OI difference
                    oi_diff = call_oi_change - put_oi_change
                    
                    # Calculate direction of change
                    if oi_diff > 0:
                        direction = 'Bullish'
                        direction_pct = (oi_diff / (abs(call_oi_change) + abs(put_oi_change))) * 100 if (abs(call_oi_change) + abs(put_oi_change)) > 0 else 0
                    elif oi_diff < 0:
                        direction = 'Bearish'
                        direction_pct = (oi_diff / (abs(call_oi_change) + abs(put_oi_change))) * 100 if (abs(call_oi_change) + abs(put_oi_change)) > 0 else 0
                    else:
                        direction = 'Neutral'
                        direction_pct = 0
                    
                    # Calculate CE/PE OI change ratio
                    ce_pe_oi_change_ratio = call_oi_change / put_oi_change if put_oi_change != 0 else np.inf
                    
                    # Calculate PCR (Put-Call Ratio)
                    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else np.inf
                    
                    # Create row for aggregated dataframe
                    row_data = {
                        'datetime': dt,
                        'Underlying_Price': group['Underlying_Price'].iloc[0],
                        'Chng_in_Call_OI': call_oi_change,
                        'Chng_in_Put_OI': put_oi_change,
                        'DIFF_in_OI': oi_diff,
                        'Direction_of_chng': direction,
                        'Direction_pct': direction_pct,
                        'Total_Call_Up': total_call_oi,
                        'Total_Put_Up': total_put_oi,
                        'CE_PE_OI_Chng': ce_pe_oi_change_ratio,
                        'Net_PCR': pcr,
                        'Sentiment': direction  # Using direction as sentiment for now
                    }
                    
                    # Append to aggregated dataframe
                    aggregated_df = pd.concat([aggregated_df, pd.DataFrame([row_data])], ignore_index=True)
                
                # Sort aggregated dataframe by datetime
                if 'datetime' in aggregated_df.columns:
                    aggregated_df.sort_values('datetime', inplace=True)
                
                # Calculate combined OI patterns for each timestamp
                # Group by datetime and strike
                datetime_strike_groups = result_df.groupby(['datetime', 'Strike'])
                
                # Initialize combined pattern columns
                result_df['Combined_OI_Pattern'] = 'Unknown'
                
                # Calculate combined patterns for each datetime and strike
                for (dt, strike), group in datetime_strike_groups:
                    # Get call and put data
                    call_data = group[group['Option_Type'].str.lower() == 'call']
                    put_data = group[group['Option_Type'].str.lower() == 'put']
                    
                    # Get OI patterns
                    call_pattern = call_data['OI_Pattern'].iloc[0] if len(call_data) > 0 else 'Unknown'
                    put_pattern = put_data['OI_Pattern'].iloc[0] if len(put_data) > 0 else 'Unknown'
                    
                    # Determine combined pattern
                    combined_pattern = 'Neutral'
                    
                    # Strong bullish patterns - From option seller's perspective
                    if (call_pattern == 'Long_Build_Up' and (put_pattern == 'Short_Build_Up' or put_pattern == 'Long_Unwinding')) or \
                       (call_pattern == 'Short_Covering' and (put_pattern == 'Short_Build_Up' or put_pattern == 'Long_Unwinding')):
                        combined_pattern = 'Strong_Bullish'
                    
                    # Mild bullish patterns
                    elif (call_pattern == 'Long_Build_Up') or \
                         (call_pattern == 'Short_Covering') or \
                         (put_pattern == 'Long_Unwinding') or \
                         (put_pattern == 'Short_Covering'):  # Put Short_Covering is bullish
                        combined_pattern = 'Mild_Bullish'
                    
                    # Strong bearish patterns - From option seller's perspective
                    elif (put_pattern == 'Long_Build_Up' and (call_pattern == 'Short_Build_Up' or call_pattern == 'Long_Unwinding')) or \
                         (put_pattern == 'Short_Covering' and (call_pattern == 'Short_Build_Up' or call_pattern == 'Long_Unwinding')):
                        combined_pattern = 'Strong_Bearish'
                    
                    # Mild bearish patterns
                    elif (put_pattern == 'Long_Build_Up') or \
                         (call_pattern == 'Long_Unwinding') or \
                         (call_pattern == 'Short_Build_Up'):
                        combined_pattern = 'Mild_Bearish'
                    
                    # Update result dataframe
                    result_df.loc[(result_df['datetime'] == dt) & (result_df['Strike'] == strike), 'Combined_OI_Pattern'] = combined_pattern
                
                # Calculate overall OI pattern for each timestamp
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize overall pattern column in aggregated dataframe
                aggregated_df['Overall_OI_Pattern'] = 'Unknown'
                
                # Calculate overall pattern for each timestamp
                for dt, group in datetime_groups:
                    # Count occurrences of each combined pattern
                    pattern_counts = group['Combined_OI_Pattern'].value_counts()
                    
                    # Determine overall pattern
                    if 'Strong_Bullish' in pattern_counts and pattern_counts['Strong_Bullish'] >= 3:
                        overall_pattern = 'Strong_Bullish'
                    elif 'Strong_Bearish' in pattern_counts and pattern_counts['Strong_Bearish'] >= 3:
                        overall_pattern = 'Strong_Bearish'
                    elif 'Mild_Bullish' in pattern_counts and pattern_counts['Mild_Bullish'] >= 3:
                        overall_pattern = 'Mild_Bullish'
                    elif 'Mild_Bearish' in pattern_counts and pattern_counts['Mild_Bearish'] >= 3:
                        overall_pattern = 'Mild_Bearish'
                    else:
                        overall_pattern = 'Neutral'
                    
                    # Update aggregated dataframe
                    aggregated_df.loc[aggregated_df['datetime'] == dt, 'Overall_OI_Pattern'] = overall_pattern
                
                # Save aggregated dataframe
                aggregated_output_path = os.path.join(self.output_dir, "trending_oi_pa_aggregated.csv")
                aggregated_df.to_csv(aggregated_output_path, index=False)
                logger.info(f"Saved aggregated trending OI with PA results to {aggregated_output_path}")
                
                # Return both detailed and aggregated results
                return result_df, aggregated_df
            
            logger.info("Calculated trending OI with price action indicators")
            
        except Exception as e:
            logger.error(f"Error calculating trending OI with price action indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df, None
    
    def visualize_trending_oi_pa(self, detailed_df, aggregated_df):
        """
        Visualize trending OI with price action indicators
        
        Args:
            detailed_df (pd.DataFrame): Detailed dataframe with OI patterns
            aggregated_df (pd.DataFrame): Aggregated dataframe with OI metrics
        """
        logger.info("Visualizing trending OI with price action indicators")
        
        # Check if we have valid dataframes
        if detailed_df is None or len(detailed_df) == 0:
            logger.error("No detailed data available for visualization")
            return
        
        if aggregated_df is None or len(aggregated_df) == 0:
            logger.error("No aggregated data available for visualization")
            return
        
        try:
            # Visualize OI changes
            if 'Chng_in_Call_OI' in aggregated_df.columns and 'Chng_in_Put_OI' in aggregated_df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(aggregated_df['datetime'], aggregated_df['Chng_in_Call_OI'], label='Change in Call OI', linewidth=2)
                plt.plot(aggregated_df['datetime'], aggregated_df['Chng_in_Put_OI'], label='Change in Put OI', linewidth=2)
                
                plt.title('Changes in Call and Put Open Interest')
                plt.xlabel('Time')
                plt.ylabel('OI Change')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                oi_changes_plot_path = os.path.join(self.output_dir, 'oi_changes.png')
                plt.savefig(oi_changes_plot_path)
                logger.info(f"Saved OI changes plot to {oi_changes_plot_path}")
                
                plt.close()
            
            # Visualize OI difference
            if 'DIFF_in_OI' in aggregated_df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(aggregated_df['datetime'], aggregated_df['DIFF_in_OI'], label='Difference in OI', linewidth=2)
                
                plt.title('Difference in OI (Call OI Change - Put OI Change)')
                plt.xlabel('Time')
                plt.ylabel('OI Difference')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                oi_diff_plot_path = os.path.join(self.output_dir, 'oi_difference.png')
                plt.savefig(oi_diff_plot_path)
                logger.info(f"Saved OI difference plot to {oi_diff_plot_path}")
                
                plt.close()
            
            # Visualize direction percentage
            if 'Direction_pct' in aggregated_df.columns:
                plt.figure(figsize=(12, 6))
                
                # Create a colormap for direction
                colors = np.where(aggregated_df['Direction_pct'] > 0, 'green', 
                                 np.where(aggregated_df['Direction_pct'] < 0, 'red', 'gray'))
                
                plt.bar(range(len(aggregated_df)), aggregated_df['Direction_pct'], color=colors)
                
                plt.title('Direction of Change Percentage')
                plt.xlabel('Time')
                plt.ylabel('Direction Percentage')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(alpha=0.3)
                
                # Save plot
                direction_plot_path = os.path.join(self.output_dir, 'direction_percentage.png')
                plt.savefig(direction_plot_path)
                logger.info(f"Saved direction percentage plot to {direction_plot_path}")
                
                plt.close()
            
            # Visualize total OI
            if 'Total_Call_Up' in aggregated_df.columns and 'Total_Put_Up' in aggregated_df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(aggregated_df['datetime'], aggregated_df['Total_Call_Up'], label='Total Call OI', linewidth=2)
                plt.plot(aggregated_df['datetime'], aggregated_df['Total_Put_Up'], label='Total Put OI', linewidth=2)
                
                plt.title('Total Call and Put Open Interest')
                plt.xlabel('Time')
                plt.ylabel('Total OI')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                total_oi_plot_path = os.path.join(self.output_dir, 'total_oi.png')
                plt.savefig(total_oi_plot_path)
                logger.info(f"Saved total OI plot to {total_oi_plot_path}")
                
                plt.close()
            
            # Visualize PCR
            if 'Net_PCR' in aggregated_df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(aggregated_df['datetime'], aggregated_df['Net_PCR'], label='Net PCR', linewidth=2)
                
                plt.title('Net Put-Call Ratio')
                plt.xlabel('Time')
                plt.ylabel('PCR')
                plt.axhline(y=1, color='r', linestyle='-', alpha=0.3, label='PCR = 1')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                pcr_plot_path = os.path.join(self.output_dir, 'net_pcr.png')
                plt.savefig(pcr_plot_path)
                logger.info(f"Saved Net PCR plot to {pcr_plot_path}")
                
                plt.close()
            
            # Visualize overall OI pattern
            if 'Overall_OI_Pattern' in aggregated_df.columns:
                plt.figure(figsize=(12, 6))
                
                # Count occurrences of each pattern
                pattern_counts = aggregated_df['Overall_OI_Pattern'].value_counts()
                
                # Create a colormap for patterns
                pattern_colors = {
                    'Strong_Bullish': 'green',
                    'Mild_Bullish': 'lightgreen',
                    'Neutral': 'gray',
                    'Mild_Bearish': 'lightcoral',
                    'Strong_Bearish': 'red',
                    'Unknown': 'blue'
                }
                
                # Plot pattern distribution
                bars = plt.bar(pattern_counts.index, pattern_counts.values, 
                              color=[pattern_colors.get(pattern, 'blue') for pattern in pattern_counts.index])
                
                plt.title('Overall OI Pattern Distribution')
                plt.xlabel('Pattern')
                plt.ylabel('Count')
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save plot
                pattern_plot_path = os.path.join(self.output_dir, 'overall_oi_pattern_distribution.png')
                plt.savefig(pattern_plot_path)
                logger.info(f"Saved overall OI pattern distribution plot to {pattern_plot_path}")
                
                plt.close()
                
                # Create a figure for pattern over time
                if 'Underlying_Price' in aggregated_df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Plot price
                    plt.plot(aggregated_df['datetime'], aggregated_df['Underlying_Price'], label='Price', alpha=0.5, color='blue')
                    
                    # Plot pattern as background color
                    x_points = range(len(aggregated_df))
                    
                    # Plot colored background for each pattern
                    for pattern, color in pattern_colors.items():
                        mask = aggregated_df['Overall_OI_Pattern'] == pattern
                        if mask.any():
                            plt.fill_between(x_points, 0, aggregated_df['Underlying_Price'].max(), where=mask, color=color, alpha=0.2, label=pattern)
                    
                    plt.title('Overall OI Pattern Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    pattern_time_plot_path = os.path.join(self.output_dir, 'overall_oi_pattern_time.png')
                    plt.savefig(pattern_time_plot_path)
                    logger.info(f"Saved overall OI pattern over time plot to {pattern_time_plot_path}")
                    
                    plt.close()
            
            # Visualize OI patterns by strike
            if 'Combined_OI_Pattern' in detailed_df.columns:
                # Get unique timestamps
                timestamps = sorted(detailed_df['datetime'].unique())
                
                # Create a figure for each timestamp
                for timestamp in timestamps[:min(5, len(timestamps))]:  # Limit to first 5 timestamps for brevity
                    plt.figure(figsize=(12, 8))
                    
                    # Get data for this timestamp
                    timestamp_data = detailed_df[detailed_df['datetime'] == timestamp]
                    
                    # Get unique strikes
                    strikes = sorted(timestamp_data['Strike'].unique())
                    
                    # Create a colormap for patterns
                    pattern_colors = {
                        'Strong_Bullish': 'green',
                        'Mild_Bullish': 'lightgreen',
                        'Neutral': 'gray',
                        'Mild_Bearish': 'lightcoral',
                        'Strong_Bearish': 'red',
                        'Unknown': 'blue'
                    }
                    
                    # Create subplots for call and put
                    plt.subplot(2, 1, 1)
                    
                    # Get call data
                    call_data = timestamp_data[timestamp_data['Option_Type'].str.lower() == 'call']
                    
                    # Plot call OI
                    for strike in strikes:
                        strike_data = call_data[call_data['Strike'] == strike]
                        if len(strike_data) > 0:
                            pattern = strike_data['Combined_OI_Pattern'].iloc[0]
                            color = pattern_colors.get(pattern, 'blue')
                            plt.bar(strike, strike_data['Open_Interest'].iloc[0], color=color, alpha=0.7)
                    
                    plt.title(f'Call OI by Strike at {timestamp}')
                    plt.xlabel('Strike')
                    plt.ylabel('Open Interest')
                    plt.grid(alpha=0.3)
                    
                    plt.subplot(2, 1, 2)
                    
                    # Get put data
                    put_data = timestamp_data[timestamp_data['Option_Type'].str.lower() == 'put']
                    
                    # Plot put OI
                    for strike in strikes:
                        strike_data = put_data[put_data['Strike'] == strike]
                        if len(strike_data) > 0:
                            pattern = strike_data['Combined_OI_Pattern'].iloc[0]
                            color = pattern_colors.get(pattern, 'blue')
                            plt.bar(strike, strike_data['Open_Interest'].iloc[0], color=color, alpha=0.7)
                    
                    plt.title(f'Put OI by Strike at {timestamp}')
                    plt.xlabel('Strike')
                    plt.ylabel('Open Interest')
                    plt.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save plot
                    strike_plot_path = os.path.join(self.output_dir, f'oi_by_strike_{timestamp}.png')
                    plt.savefig(strike_plot_path)
                    logger.info(f"Saved OI by strike plot for {timestamp} to {strike_plot_path}")
                    
                    plt.close()
            
            logger.info("Completed visualization of trending OI with price action indicators")
            
        except Exception as e:
            logger.error(f"Error visualizing trending OI with price action indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_trending_oi_pa(self):
        """
        Test trending OI with price action analysis
        
        Returns:
            tuple: (detailed_df, aggregated_df) - Detailed and aggregated dataframes with trending OI with PA indicators
        """
        logger.info("Testing trending OI with price action analysis")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None, None
        
        # Calculate trending OI with PA indicators
        detailed_df, aggregated_df = self.calculate_trending_oi_pa(df)
        
        # Save detailed results
        if detailed_df is not None:
            detailed_output_path = os.path.join(self.output_dir, "trending_oi_pa_detailed.csv")
            detailed_df.to_csv(detailed_output_path, index=False)
            logger.info(f"Saved detailed trending OI with PA results to {detailed_output_path}")
        
        # Visualize results
        if detailed_df is not None and aggregated_df is not None:
            self.visualize_trending_oi_pa(detailed_df, aggregated_df)
        
        # Log summary statistics
        if aggregated_df is not None and 'Overall_OI_Pattern' in aggregated_df.columns:
            pattern_counts = aggregated_df['Overall_OI_Pattern'].value_counts()
            logger.info(f"Overall OI Pattern distribution: {pattern_counts.to_dict()}")
        
        logger.info("Trending OI with price action analysis testing completed")
        
        return detailed_df, aggregated_df

def main():
    """
    Main function to run the trending OI with price action analysis testing
    """
    logger.info("Starting trending OI with price action analysis testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/trending_oi_pa',
        'strikes_above_atm': 7,
        'strikes_below_atm': 7,
        'lookback_period': 5,
        'time_interval': '5min'
    }
    
    # Create trending OI with PA tester
    tester = TrendingOIPATester(config)
    
    # Test trending OI with PA
    detailed_df, aggregated_df = tester.test_trending_oi_pa()
    
    logger.info("Trending OI with price action analysis testing completed")
    
    return detailed_df, aggregated_df

if __name__ == "__main__":
    main()
