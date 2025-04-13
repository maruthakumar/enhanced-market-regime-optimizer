import os
import pandas as pd
import numpy as np
import glob
import logging
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_option_data(directory):
    """Load option data from CSV files in directory."""
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        logger.error(f"No CSV files found in {directory}")
        return None
    
    logger.info(f"Found {len(all_files)} CSV files")
    
    # Load all files into a single dataframe
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def fix_date_column(df):
    """Fix date column to ensure it's in a proper datetime format."""
    # First check the date column
    logger.info(f"Date column dtype: {df['date'].dtype}")
    
    # Sample of date values
    logger.info(f"Date sample values: {df['date'].head(5).tolist()}")
    
    # Convert to datetime with proper error handling
    try:
        if pd.api.types.is_numeric_dtype(df['date']):
            # First, handle any non-finite values
            df = df[pd.notna(df['date'])].copy()
            
            # Convert numeric to string (handling possible decimals)
            df['date_str'] = df['date'].astype(int).astype(str)
            
            # Check the string length and make sure it's 6 digits
            sample_len = df['date_str'].str.len().value_counts()
            logger.info(f"Date string lengths: {sample_len.to_dict()}")
            
            # For dates like 230403 (April 3, 2023), parse directly
            df['date'] = pd.to_datetime(df['date_str'], format='%y%m%d', errors='coerce')
            
            # Check if date parsing worked
            valid_dates = pd.notna(df['date'])
            logger.info(f"Successfully parsed {valid_dates.sum()} dates out of {len(df)}")
            
            # Drop rows with invalid dates
            df = df[valid_dates].copy()
            
            # Show sample of converted dates
            if not df.empty:
                logger.info(f"Sample converted dates: {df['date'].head(3)}")
        else:
            # Try standard datetime parsing
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Drop rows with NaT dates
            df = df[pd.notna(df['date'])].copy()
    
    except Exception as e:
        logger.error(f"Error converting dates: {str(e)}")
        # Try a different approach if the first one fails
        try:
            # Try parsing with coercion to handle non-finite values
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Drop rows with NaT dates
            df = df[pd.notna(df['date'])].copy()
        except Exception as e2:
            logger.error(f"Second attempt at date conversion failed: {str(e2)}")
    
    return df

def filter_relevant_options(df):
    """Filter options with delta between 0.5 and 0.1 for calls, -0.5 to -0.1 for puts."""
    if df is None or df.empty:
        return None
    
    # Check if required columns exist
    required_cols = ['date', 'time', 'expiry', 'call_delta', 'call_vega', 'call_theta', 'call_gamma',
                     'put_delta', 'put_vega', 'put_theta', 'put_gamma', 'underlying_price']
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing required columns: {missing}")
        return None
    
    # Fix date column
    df = fix_date_column(df)
    
    # Make sure date is actually a datetime type
    if not pd.api.types.is_datetime64_dtype(df['date']):
        logger.error("Date column is not a datetime type after conversion")
        return None
    
    # Filter relevant options based on delta
    call_options = df[(df['call_delta'] <= 0.5) & (df['call_delta'] >= 0.1)].copy()
    put_options = df[(df['put_delta'] >= -0.5) & (df['put_delta'] <= -0.1)].copy()
    
    logger.info(f"Filtered {len(call_options)} call options and {len(put_options)} put options")
    
    # Display unique dates
    call_dates = sorted(call_options['date'].dt.date.unique())
    put_dates = sorted(put_options['date'].dt.date.unique())
    
    logger.info(f"Data covers {len(call_dates)} unique dates for calls")
    logger.info(f"Data covers {len(put_dates)} unique dates for puts")
    
    if call_dates:
        logger.info(f"Sample call dates: {call_dates[:5]}")
    if put_dates:
        logger.info(f"Sample put dates: {put_dates[:5]}")
    
    # Check time column format and convert if needed
    time_samples = call_options['time'].head(5).tolist()
    logger.info(f"Time column format samples: {time_samples}")
    
    return {
        'calls': call_options,
        'puts': put_options
    }

def calculate_daily_sentiment(options, output_file):
    """Calculate Greek sentiment for each day and save results."""
    if options is None:
        return False
    
    calls = options['calls']
    puts = options['puts']
    
    if calls.empty or puts.empty:
        logger.warning("No call or put data available")
        return False
    
    # Group by date to calculate daily sentiment
    results = []
    
    # Get valid dates (exclude NaT)
    valid_call_dates = pd.to_datetime(calls['date']).dt.date.dropna().unique()
    valid_put_dates = pd.to_datetime(puts['date']).dt.date.dropna().unique()
    
    # Find common dates between calls and puts
    common_dates = sorted(set(valid_call_dates).intersection(set(valid_put_dates)))
    
    logger.info(f"Found {len(common_dates)} dates with both call and put data")
    if common_dates:
        logger.info(f"Sample dates: {common_dates[:5]}")
    else:
        logger.error("No common dates between call and put data")
        return False
    
    # Process each day
    for date in common_dates:
        # Filter data for this date
        day_calls = calls[pd.to_datetime(calls['date']).dt.date == date]
        day_puts = puts[pd.to_datetime(puts['date']).dt.date == date]
        
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"Processing date: {date_str}")
        
        # Get unique times for this day (sorted)
        call_times = set(day_calls['time'])
        put_times = set(day_puts['time'])
        common_times = sorted(call_times.intersection(put_times))
        
        logger.info(f"Found {len(common_times)} common times for date {date_str}")
        
        if not common_times:
            logger.warning(f"No common time data for {date_str}")
            continue
        
        # First time as baseline
        baseline_time = common_times[0]
        logger.info(f"Using baseline time: {baseline_time}")
        
        # Calculate baseline Greeks at market open
        baseline_calls = day_calls[day_calls['time'] == baseline_time]
        baseline_puts = day_puts[day_puts['time'] == baseline_time]
        
        if baseline_calls.empty or baseline_puts.empty:
            logger.warning(f"No baseline data for {date_str} at time {baseline_time}")
            continue
        
        # Initialize baseline values
        baseline_call_vega = baseline_calls['call_vega'].sum()
        baseline_call_delta = baseline_calls['call_delta'].sum()
        baseline_call_theta = baseline_calls['call_theta'].sum()
        
        baseline_put_vega = baseline_puts['put_vega'].sum()
        baseline_put_delta = baseline_puts['put_delta'].sum()
        baseline_put_theta = baseline_puts['put_theta'].sum()
        
        logger.info(f"Baseline values - Call: vega={baseline_call_vega:.2f}, delta={baseline_call_delta:.2f} | "
                    f"Put: vega={baseline_put_vega:.2f}, delta={baseline_put_delta:.2f}")
        
        # Calculate sentiment for each time after baseline
        for time in common_times[1:]:
            time_calls = day_calls[day_calls['time'] == time]
            time_puts = day_puts[day_puts['time'] == time]
            
            if time_calls.empty or time_puts.empty:
                logger.debug(f"Missing data for time {time}")
                continue
            
            # Current Greek values
            curr_call_vega = time_calls['call_vega'].sum()
            curr_call_delta = time_calls['call_delta'].sum()
            curr_call_theta = time_calls['call_theta'].sum()
            
            curr_put_vega = time_puts['put_vega'].sum()
            curr_put_delta = time_puts['put_delta'].sum()
            curr_put_theta = time_puts['put_theta'].sum()
            
            # Calculate changes from baseline
            delta_call_vega = curr_call_vega - baseline_call_vega
            delta_call_delta = curr_call_delta - baseline_call_delta
            delta_call_theta = curr_call_theta - baseline_call_theta
            
            delta_put_vega = curr_put_vega - baseline_put_vega
            delta_put_delta = curr_put_delta - baseline_put_delta
            delta_put_theta = curr_put_theta - baseline_put_theta
            
            # Calculate sentiment
            # According to docs:
            # - If SumVegaPut is Large Negative: Bearish
            # - If SumVegaPut is Large Positive: Bullish
            # - If SumVegaCall is Large Negative: Bullish
            # - If SumVegaCall is Large Positive: Bearish
            
            # Combined vega sentiment (equal weight for call and put)
            vega_sentiment = -delta_call_vega + delta_put_vega
            
            # Delta sentiment (Delta changes can confirm direction)
            delta_sentiment = delta_call_delta - delta_put_delta
            
            # Combined sentiment (can be adjusted with weights)
            combined_sentiment = (vega_sentiment + delta_sentiment) / 2
            
            # Determine sentiment label
            sentiment = "Neutral"
            if combined_sentiment > 10:
                sentiment = "Bullish"
            elif combined_sentiment < -10:
                sentiment = "Bearish"
            
            # Store result
            results.append({
                'Date': date_str,
                'Time': time,
                'Call_Vega_Change': delta_call_vega,
                'Put_Vega_Change': delta_put_vega,
                'Call_Delta_Change': delta_call_delta,
                'Put_Delta_Change': delta_put_delta,
                'Call_Theta_Change': delta_call_theta,
                'Put_Theta_Change': delta_put_theta,
                'Vega_Sentiment': vega_sentiment,
                'Delta_Sentiment': delta_sentiment,
                'Combined_Sentiment': combined_sentiment,
                'Sentiment': sentiment
            })
    
    # Create results dataframe
    if results:
        logger.info(f"Generated {len(results)} sentiment data points")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved sentiment results to {output_file}")
        return True
    else:
        logger.warning("No sentiment results calculated")
        return False

def main():
    parser = argparse.ArgumentParser(description='Calculate Greek sentiment indicator from option data')
    parser.add_argument('--input-dir', default='data/market_data/formatted',
                        help='Directory containing formatted option data')
    parser.add_argument('--output-file', default='data/market_data/greek_sentiment.csv',
                        help='Output file for sentiment results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load option data
    df = load_option_data(args.input_dir)
    if df is None:
        logger.error("Failed to load option data")
        return
    
    # Filter relevant options
    filtered_options = filter_relevant_options(df)
    if filtered_options is None:
        logger.error("Failed to filter relevant options")
        return
    
    # Calculate sentiment
    success = calculate_daily_sentiment(filtered_options, args.output_file)
    
    if success:
        logger.info("Greek sentiment calculation completed successfully")
    else:
        logger.error("Failed to calculate Greek sentiment")

if __name__ == "__main__":
    main() 