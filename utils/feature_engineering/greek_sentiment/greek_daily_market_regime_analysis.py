"""
Greek Daily Market Regime Analysis Script

This script analyzes Greek sentiment data from options and provides daily market regime classifications
with signals for regime transitions, specifically designed for day traders.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import json
import sys
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Greek-specific core modules
from greek_data_processor import process_data_efficiently, calculate_greek_sentiment_efficiently
from greek_dynamic_weight_adjuster import GreekDynamicWeightAdjuster
from greek_regime_classifier import GreekRegimeClassifier
from greek_market_regime import process_greek_sentiment

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Daily Greek market regime analysis for day traders')
    
    parser.add_argument('--data-dir', type=str, default='../data/market_data',
                        help='Directory containing market data files')
    
    parser.add_argument('--output-dir', type=str, default='../output/daily_analysis',
                        help='Directory to save analysis results')
    
    parser.add_argument('--config-file', type=str, default='../config/day_trader_config.ini',
                        help='Configuration file path')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for analysis (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for analysis (YYYY-MM-DD)')
    
    parser.add_argument('--lookback-days', type=int, default=5,
                        help='Number of days to look back for regime classification')
    
    parser.add_argument('--intraday', action='store_true',
                        help='Enable intraday analysis (requires timestamp in data)')
    
    parser.add_argument('--intraday-intervals', type=int, default=3,
                        help='Number of intraday intervals for analysis')
    
    parser.add_argument('--time-granularity', type=int, default=0,
                        help='Time granularity in minutes (0 for predefined intervals)')
    
    return parser.parse_args()

def create_day_trader_config(args):
    """
    Create configuration for day trader analysis.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        ConfigParser: Configuration for day trader analysis
    """
    config = configparser.ConfigParser()
    
    # Data processing section
    config.add_section('data_processing')
    config.set('data_processing', 'market_data_dir', args.data_dir)
    config.set('data_processing', 'chunk_size', '50000')
    config.set('data_processing', 'use_parallel', 'true')
    config.set('data_processing', 'num_processes', '2')
    config.set('data_processing', 'memory_limit_mb', '1000')
    
    # Greek sentiment section
    config.add_section('greek_sentiment')
    config.set('greek_sentiment', 'enable_dynamic_weights', 'true')
    config.set('greek_sentiment', 'delta_weight', '1.2')
    config.set('greek_sentiment', 'vega_weight', '1.5')
    config.set('greek_sentiment', 'theta_weight', '0.3')
    config.set('greek_sentiment', 'sentiment_threshold_bullish', '5.0')  # More sensitive for day trading
    config.set('greek_sentiment', 'sentiment_threshold_bearish', '-5.0')  # More sensitive for day trading
    config.set('greek_sentiment', 'weight_history_window', str(args.lookback_days))
    config.set('greek_sentiment', 'weight_learning_rate', '0.1')  # Faster adaptation for day trading
    config.set('greek_sentiment', 'min_weight', '0.1')
    config.set('greek_sentiment', 'max_weight', '3.0')
    
    # Market regime section
    config.add_section('market_regime')
    config.set('market_regime', 'market_data_dir', args.data_dir)
    config.set('market_regime', 'enable_greek_sentiment', 'true')
    config.set('market_regime', 'num_regimes', '5')
    config.set('market_regime', 'use_clustering', 'true')
    config.set('market_regime', 'adaptive_thresholds', 'true')
    config.set('market_regime', 'regime_lookback_window', str(args.lookback_days))
    config.set('market_regime', 'strong_bearish_threshold', '-8.0')
    config.set('market_regime', 'bearish_threshold', '-2.5')
    config.set('market_regime', 'bullish_threshold', '2.5')
    config.set('market_regime', 'strong_bullish_threshold', '8.0')
    
    # Day trader specific section
    config.add_section('day_trader')
    config.set('day_trader', 'intraday_analysis', str(args.intraday).lower())
    config.set('day_trader', 'intraday_intervals', str(args.intraday_intervals))
    config.set('day_trader', 'time_granularity', str(args.time_granularity))
    config.set('day_trader', 'highlight_regime_transitions', 'true')
    config.set('day_trader', 'alert_on_transition', 'true')
    
    # Output section
    config.add_section('output')
    config.set('output', 'base_dir', args.output_dir)
    config.set('output', 'log_dir', os.path.join(args.output_dir, 'logs'))
    
    # Save configuration if config file path is provided
    if args.config_file:
        os.makedirs(os.path.dirname(args.config_file), exist_ok=True)
        with open(args.config_file, 'w') as f:
            config.write(f)
    
    return config

def load_and_process_data(config, args):
    """
    Load and process market data for analysis.
    
    Args:
        config (ConfigParser): Configuration
        args (argparse.Namespace): Command line arguments
        
    Returns:
        DataFrame: Processed data with sentiment indicators
    """
    logger.info("Loading and processing market data")
    
    data_dir = args.data_dir
    
    # Find data files
    data_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))
    
    if not data_files:
        logger.error(f"No data files found in {data_dir}")
        return None
    
    logger.info(f"Found {len(data_files)} data files")
    
    # Process data files
    output_dir = os.path.join(args.output_dir, 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    processed_file = os.path.join(output_dir, 'processed_data.csv')
    processed_data = process_data_efficiently(data_files, config, processed_file)
    
    if processed_data is None:
        # Try to load processed data from file
        if os.path.exists(processed_file):
            logger.info(f"Loading processed data from {processed_file}")
            processed_data = pd.read_csv(processed_file)
        else:
            logger.error("Failed to process data and no processed data file found")
            return None
    
    # Calculate Greek sentiment
    sentiment_file = os.path.join(output_dir, 'greek_sentiment.csv')
    
    try:
        sentiment_data = calculate_greek_sentiment_efficiently(processed_data, config, sentiment_file)
        logger.info(f"Generated sentiment data with {len(sentiment_data)} rows")
    except Exception as e:
        logger.error(f"Error calculating sentiment: {str(e)}")
        
        # Try to load sentiment data from file
        if os.path.exists(sentiment_file):
            logger.info(f"Loading sentiment data from {sentiment_file}")
            sentiment_data = pd.read_csv(sentiment_file)
        else:
            logger.error("Failed to calculate sentiment and no sentiment data file found")
            return None
    
    # Process Greek sentiment
    processed_sentiment = process_greek_sentiment(sentiment_data, config)
    
    # Filter by date range if specified
    if args.start_date or args.end_date:
        # Find date column
        date_col = None
        for col in ['Date', 'date', 'datetime', 'timestamp']:
            if col in processed_sentiment.columns:
                date_col = col
                break
        
        if date_col:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(processed_sentiment[date_col]):
                processed_sentiment[date_col] = pd.to_datetime(processed_sentiment[date_col])
            
            # Filter by start date
            if args.start_date:
                start_date = pd.to_datetime(args.start_date)
                processed_sentiment = processed_sentiment[processed_sentiment[date_col] >= start_date]
            
            # Filter by end date
            if args.end_date:
                end_date = pd.to_datetime(args.end_date)
                processed_sentiment = processed_sentiment[processed_sentiment[date_col] <= end_date]
            
            logger.info(f"Filtered data to {len(processed_sentiment)} rows based on date range")
        else:
            logger.warning("No date column found, cannot filter by date range")
    
    return processed_sentiment

def classify_daily_regimes(data, config):
    """
    Classify market regimes on a daily basis.
    
    Args:
        data (DataFrame): Processed data with sentiment indicators
        config (ConfigParser): Configuration
        
    Returns:
        DataFrame: Data with daily regime classifications
    """
    logger.info("Classifying daily market regimes")
    
    # Create classifier
    classifier = GreekRegimeClassifier(config)
    
    # Classify regimes
    classified_data = classifier.classify_regimes(data)
    
    # Add regime transition indicators
    classified_data = add_regime_transition_indicators(classified_data)
    
    return classified_data

def add_regime_transition_indicators(data):
    """
    Add regime transition indicators to the data.
    
    Args:
        data (DataFrame): Data with regime classifications
        
    Returns:
        DataFrame: Data with regime transition indicators
    """
    if 'Market_Regime' not in data.columns:
        logger.warning("Market_Regime column not found, cannot add transition indicators")
        return data
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Add regime shift indicator if not already present
    if 'Regime_Shift' not in result.columns:
        result['Regime_Shift'] = False
        
        # Check for regime changes
        if len(result) > 1:
            result.loc[result['Market_Regime'].shift() != result['Market_Regime'], 'Regime_Shift'] = True
    
    # Add regime transition type
    result['Transition_Type'] = 'None'
    
    # Define regime strength order
    regime_strength = {
        'Strong Bearish': -2,
        'Bearish': -1,
        'Neutral': 0,
        'Bullish': 1,
        'Strong Bullish': 2
    }
    
    # Add numeric regime strength
    if all(regime in regime_strength for regime in result['Market_Regime'].unique()):
        result['Regime_Strength'] = result['Market_Regime'].map(regime_strength)
        
        # Determine transition type
        for i in range(1, len(result)):
            if result.iloc[i]['Regime_Shift']:
                prev_strength = result.iloc[i-1]['Regime_Strength']
                curr_strength = result.iloc[i]['Regime_Strength']
                
                if curr_strength > prev_strength:
                    result.iloc[i, result.columns.get_loc('Transition_Type')] = 'Bullish'
                elif curr_strength < prev_strength:
                    result.iloc[i, result.columns.get_loc('Transition_Type')] = 'Bearish'
                else:
                    result.iloc[i, result.columns.get_loc('Transition_Type')] = 'Neutral'
    
    # Add trading signal based on transition
    result['Trading_Signal'] = 'Hold'
    
    # Generate trading signals based on regime transitions
    for i in range(1, len(result)):
        if result.iloc[i]['Regime_Shift']:
            transition = result.iloc[i]['Transition_Type']
            
            if transition == 'Bullish':
                result.iloc[i, result.columns.get_loc('Trading_Signal')] = 'Buy'
            elif transition == 'Bearish':
                result.iloc[i, result.columns.get_loc('Trading_Signal')] = 'Sell'
    
    return result

def analyze_intraday_regimes(data, config):
    """
    Analyze market regimes on an intraday basis.
    
    Args:
        data (DataFrame): Processed data with sentiment indicators
        config (ConfigParser): Configuration
        
    Returns:
        DataFrame: Data with intraday regime analysis
    """
    logger.info("Analyzing intraday market regimes")
    
    # Check if we have timestamp information
    timestamp_col = None
    for col in ['timestamp', 'datetime', 'Date', 'date']:
        if col in data.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        logger.warning("No timestamp column found, cannot perform intraday analysis")
        return data
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(data[timestamp_col]):
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    # Get time granularity
    time_granularity = config.getint('day_trader', 'time_granularity', fallback=0)
    
    # If time granularity is specified, use it instead of predefined intervals
    if time_granularity > 0:
        logger.info(f"Using {time_granularity}-minute granularity for intraday analysis")
        
        # Add time period based on granularity
        data['Minute'] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
        data['Time_Period'] = (data['Minute'] // time_granularity) * time_granularity
        data['Time_Period'] = data['Time_Period'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")
        
        # Group by date and time period
        grouped = data.groupby([data[timestamp_col].dt.date, 'Time_Period'])
    else:
        # Get intraday intervals
        intervals = config.getint('day_trader', 'intraday_intervals', fallback=3)
        
        # Add time of day indicator
        data['Hour'] = data[timestamp_col].dt.hour
        
        # Define time periods based on trading hours (assuming 9:30 AM to 4:00 PM)
        if intervals == 3:
            # Morning, Midday, Afternoon
            data['Time_Period'] = 'Midday'
            data.loc[data['Hour'] < 11, 'Time_Period'] = 'Morning'
            data.loc[data['Hour'] >= 14, 'Time_Period'] = 'Afternoon'
        elif intervals == 4:
            # Early Morning, Late Morning, Early Afternoon, Late Afternoon
            data['Time_Period'] = 'Late Morning'
            data.loc[data['Hour'] < 10, 'Time_Period'] = 'Early Morning'
            data.loc[(data['Hour'] >= 12) & (data['Hour'] < 14), 'Time_Period'] = 'Early Afternoon'
            data.loc[data['Hour'] >= 14, 'Time_Period'] = 'Late Afternoon'
        else:
            # Custom intervals
            data['Time_Period'] = pd.cut(
                data['Hour'],
                bins=intervals,
                labels=[f'Period {i+1}' for i in range(intervals)]
            )
        
        # Group by date and time period
        grouped = data.groupby([data[timestamp_col].dt.date, 'Time_Period'])
    
    # Create empty list to store results
    results = []
    
    # Process each group
    for (date, period), group in grouped:
        # Calculate average sentiment for the period
        avg_sentiment = group['Combined_Sentiment'].mean()
        
        # Get most common regime for the period
        if 'Market_Regime' in group.columns:
            regime_counts = group['Market_Regime'].value_counts()
            most_common_regime = regime_counts.index[0]
            regime_confidence = regime_counts.iloc[0] / len(group)
        else:
            most_common_regime = 'Unknown'
            regime_confidence = 0.0
        
        # Create result row
        result = {
            'Date': date,
            'Time_Period': period,
            'Avg_Sentiment': avg_sentiment,
            'Market_Regime': most_common_regime,
            'Regime_Confidence': regime_confidence,
            'Data_Points': len(group)
        }
        
        results.append(result)
    
    # Convert results to DataFrame
    intraday_data = pd.DataFrame(results)
    
    # Sort by date and time period
    intraday_data = intraday_data.sort_values(['Date', 'Time_Period'])
    
    # Add regime transition indicators
    intraday_data = add_regime_transition_indicators(intraday_data)
    
    return intraday_data

def generate_daily_report(daily_data, intraday_data, config, args):
    """
    Generate a daily report for day traders.
    
    Args:
        daily_data (DataFrame): Data with daily regime classifications
        intraday_data (DataFrame): Data with intraday regime analysis
        config (ConfigParser): Configuration
        args (argparse.Namespace): Command line arguments
        
    Returns:
        str: Path to the report file
    """
    logger.info("Generating daily report for day traders")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create report file
    report_file = os.path.join(output_dir, f'greek_daily_report_{timestamp}.html')
    
    # Generate HTML report
    with open(report_file, 'w') as f:
        # Write HTML header
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Greek Market Regime Report for Day Traders</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333366; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .bullish { color: green; font-weight: bold; }
                .bearish { color: red; font-weight: bold; }
                .neutral { color: gray; }
                .transition { background-color: #ffffcc; }
                .buy { color: green; font-weight: bold; }
                .sell { color: red; font-weight: bold; }
                .hold { color: gray; }
                .chart { width: 100%; height: 400px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Greek Market Regime Report for Day Traders</h1>
            <p>Generated on: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        ''')
        
        # Write summary section
        f.write('''
            <h2>Summary</h2>
            <p>This report provides market regime analysis based on Greek sentiment indicators (Delta, Vega, Theta) for day traders.</p>
        ''')
        
        # Write daily regime section
        f.write('''
            <h2>Daily Greek Market Regimes</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Market Regime</th>
                    <th>Sentiment</th>
                    <th>Regime Shift</th>
                    <th>Transition Type</th>
                    <th>Trading Signal</th>
                </tr>
        ''')
        
        # Find date column
        date_col = None
        for col in ['Date', 'date', 'datetime', 'timestamp']:
            if col in daily_data.columns:
                date_col = col
                break
        
        if date_col:
            # Sort by date
            daily_data = daily_data.sort_values(date_col)
            
            # Write daily data rows
            for _, row in daily_data.iterrows():
                date_str = str(row[date_col])
                regime = row['Market_Regime'] if 'Market_Regime' in row else 'Unknown'
                sentiment = f"{row['Combined_Sentiment']:.2f}" if 'Combined_Sentiment' in row else 'N/A'
                
                regime_shift = row['Regime_Shift'] if 'Regime_Shift' in row else False
                transition_type = row['Transition_Type'] if 'Transition_Type' in row else 'None'
                trading_signal = row['Trading_Signal'] if 'Trading_Signal' in row else 'Hold'
                
                # Determine CSS classes
                regime_class = regime.lower().replace(' ', '_')
                signal_class = trading_signal.lower()
                row_class = 'transition' if regime_shift else ''
                
                f.write(f'''
                    <tr class="{row_class}">
                        <td>{date_str}</td>
                        <td class="{regime_class}">{regime}</td>
                        <td>{sentiment}</td>
                        <td>{"Yes" if regime_shift else "No"}</td>
                        <td>{transition_type}</td>
                        <td class="{signal_class}">{trading_signal}</td>
                    </tr>
                ''')
        
        f.write('''
            </table>
        ''')
        
        # Write intraday regime section if available
        if intraday_data is not None and not intraday_data.empty:
            f.write('''
                <h2>Intraday Greek Market Regimes</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Time Period</th>
                        <th>Market Regime</th>
                        <th>Sentiment</th>
                        <th>Confidence</th>
                        <th>Regime Shift</th>
                        <th>Trading Signal</th>
                    </tr>
            ''')
            
            # Write intraday data rows
            for _, row in intraday_data.iterrows():
                date_str = str(row['Date'])
                time_period = row['Time_Period']
                regime = row['Market_Regime'] if 'Market_Regime' in row else 'Unknown'
                sentiment = f"{row['Avg_Sentiment']:.2f}" if 'Avg_Sentiment' in row else 'N/A'
                confidence = f"{row['Regime_Confidence']*100:.1f}%" if 'Regime_Confidence' in row else 'N/A'
                
                regime_shift = row['Regime_Shift'] if 'Regime_Shift' in row else False
                trading_signal = row['Trading_Signal'] if 'Trading_Signal' in row else 'Hold'
                
                # Determine CSS classes
                regime_class = regime.lower().replace(' ', '_')
                signal_class = trading_signal.lower()
                row_class = 'transition' if regime_shift else ''
                
                f.write(f'''
                    <tr class="{row_class}">
                        <td>{date_str}</td>
                        <td>{time_period}</td>
                        <td class="{regime_class}">{regime}</td>
                        <td>{sentiment}</td>
                        <td>{confidence}</td>
                        <td>{"Yes" if regime_shift else "No"}</td>
                        <td class="{signal_class}">{trading_signal}</td>
                    </tr>
                ''')
            
            f.write('''
                </table>
            ''')
        
        # Write trading recommendations section
        f.write('''
            <h2>Greek Trading Recommendations</h2>
            <table>
                <tr>
                    <th>Market Regime</th>
                    <th>Recommended Strategy</th>
                </tr>
                <tr>
                    <td class="strong_bullish">Strong Bullish</td>
                    <td>Aggressive long positions, focus on call options with higher delta</td>
                </tr>
                <tr>
                    <td class="bullish">Bullish</td>
                    <td>Long positions, balanced call options, consider bull spreads</td>
                </tr>
                <tr>
                    <td class="neutral">Neutral</td>
                    <td>Range-bound strategies, iron condors, calendar spreads</td>
                </tr>
                <tr>
                    <td class="bearish">Bearish</td>
                    <td>Short positions, balanced put options, consider bear spreads</td>
                </tr>
                <tr>
                    <td class="strong_bearish">Strong Bearish</td>
                    <td>Aggressive short positions, focus on put options with higher delta</td>
                </tr>
            </table>
        ''')
        
        # Write HTML footer
        f.write('''
        </body>
        </html>
        ''')
    
    logger.info(f"Greek daily report saved to {report_file}")
    return report_file

def create_visualization(daily_data, intraday_data, config, args):
    """
    Create visualizations for day traders.
    
    Args:
        daily_data (DataFrame): Data with daily regime classifications
        intraday_data (DataFrame): Data with intraday regime analysis
        config (ConfigParser): Configuration
        args (argparse.Namespace): Command line arguments
        
    Returns:
        list: Paths to visualization files
    """
    logger.info("Creating Greek visualizations for day traders")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # List to store visualization files
    visualization_files = []
    
    # Find date column
    date_col = None
    for col in ['Date', 'date', 'datetime', 'timestamp']:
        if col in daily_data.columns:
            date_col = col
            break
    
    if date_col:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(daily_data[date_col]):
            daily_data[date_col] = pd.to_datetime(daily_data[date_col])
        
        # Sort by date
        daily_data = daily_data.sort_values(date_col)
        
        # Create regime distribution visualization
        if 'Market_Regime' in daily_data.columns:
            plt.figure(figsize=(12, 6))
            
            regime_counts = daily_data['Market_Regime'].value_counts()
            sns.barplot(x=regime_counts.index, y=regime_counts.values)
            
            plt.title('Greek Market Regime Distribution')
            plt.xlabel('Regime')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save visualization
            regime_dist_file = os.path.join(output_dir, f'greek_regime_distribution_{timestamp}.png')
            plt.savefig(regime_dist_file)
            plt.close()
            
            visualization_files.append(regime_dist_file)
            logger.info(f"Greek regime distribution visualization saved to {regime_dist_file}")
        
        # Create sentiment trend visualization
        if 'Combined_Sentiment' in daily_data.columns:
            plt.figure(figsize=(14, 7))
            
            # Plot sentiment
            plt.plot(daily_data[date_col], daily_data['Combined_Sentiment'], 'b-', label='Combined Greek Sentiment')
            
            # Add horizontal lines for thresholds
            bullish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bullish', fallback=5.0)
            bearish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bearish', fallback=-5.0)
            
            plt.axhline(y=bullish_threshold, color='g', linestyle='--', alpha=0.5, label='Bullish Threshold')
            plt.axhline(y=bearish_threshold, color='r', linestyle='--', alpha=0.5, label='Bearish Threshold')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Highlight regime transitions
            if 'Regime_Shift' in daily_data.columns:
                transitions = daily_data[daily_data['Regime_Shift']]
                if not transitions.empty:
                    plt.scatter(
                        transitions[date_col],
                        transitions['Combined_Sentiment'],
                        color='r',
                        s=100,
                        marker='^',
                        label='Regime Transition'
                    )
            
            plt.title('Greek Sentiment Trend')
            plt.xlabel('Date')
            plt.ylabel('Sentiment Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save visualization
            sentiment_trend_file = os.path.join(output_dir, f'greek_sentiment_trend_{timestamp}.png')
            plt.savefig(sentiment_trend_file)
            plt.close()
            
            visualization_files.append(sentiment_trend_file)
            logger.info(f"Greek sentiment trend visualization saved to {sentiment_trend_file}")
        
        # Create regime transitions visualization
        if 'Market_Regime' in daily_data.columns:
            plt.figure(figsize=(14, 7))
            
            # Create a numeric mapping for regimes
            regimes = daily_data['Market_Regime'].unique()
            regime_to_num = {regime: i for i, regime in enumerate(sorted(regimes))}
            
            # Convert regimes to numeric values
            regime_numeric = daily_data['Market_Regime'].map(regime_to_num)
            
            # Plot regime transitions
            plt.plot(daily_data[date_col], regime_numeric, 'b-')
            
            # Highlight regime transitions
            if 'Regime_Shift' in daily_data.columns:
                transitions = daily_data[daily_data['Regime_Shift']]
                if not transitions.empty:
                    plt.scatter(
                        transitions[date_col],
                        transitions['Market_Regime'].map(regime_to_num),
                        color='r',
                        s=100,
                        marker='^'
                    )
            
            # Set y-ticks to regime names
            plt.yticks(
                [regime_to_num[regime] for regime in sorted(regimes)],
                sorted(regimes)
            )
            
            plt.title('Greek Market Regime Transitions')
            plt.xlabel('Date')
            plt.ylabel('Market Regime')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save visualization
            regime_trans_file = os.path.join(output_dir, f'greek_regime_transitions_{timestamp}.png')
            plt.savefig(regime_trans_file)
            plt.close()
            
            visualization_files.append(regime_trans_file)
            logger.info(f"Greek regime transitions visualization saved to {regime_trans_file}")
    
    # Create intraday visualization if available
    if intraday_data is not None and not intraday_data.empty:
        # Convert Date to datetime if not already
        if not pd.api.types.is_datetime64_dtype(intraday_data['Date']):
            intraday_data['Date'] = pd.to_datetime(intraday_data['Date'])
        
        # Create intraday heatmap
        if 'Avg_Sentiment' in intraday_data.columns and 'Time_Period' in intraday_data.columns:
            # Pivot data for heatmap
            pivot_data = intraday_data.pivot(index='Date', columns='Time_Period', values='Avg_Sentiment')
            
            plt.figure(figsize=(14, 8))
            
            # Create heatmap
            sns.heatmap(
                pivot_data,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                linewidths=0.5
            )
            
            plt.title('Greek Intraday Sentiment Heatmap')
            plt.tight_layout()
            
            # Save visualization
            intraday_heatmap_file = os.path.join(output_dir, f'greek_intraday_heatmap_{timestamp}.png')
            plt.savefig(intraday_heatmap_file)
            plt.close()
            
            visualization_files.append(intraday_heatmap_file)
            logger.info(f"Greek intraday heatmap visualization saved to {intraday_heatmap_file}")
    
    # Create dashboard visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Regime distribution
    if 'Market_Regime' in daily_data.columns:
        plt.subplot(2, 2, 1)
        regime_counts = daily_data['Market_Regime'].value_counts()
        sns.barplot(x=regime_counts.index, y=regime_counts.values)
        plt.title('Greek Market Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    # Plot 2: Sentiment trend
    if 'Combined_Sentiment' in daily_data.columns and date_col:
        plt.subplot(2, 2, 2)
        plt.plot(daily_data[date_col], daily_data['Combined_Sentiment'], 'b-')
        
        # Add horizontal lines for thresholds
        bullish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bullish', fallback=5.0)
        bearish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bearish', fallback=-5.0)
        
        plt.axhline(y=bullish_threshold, color='g', linestyle='--', alpha=0.5)
        plt.axhline(y=bearish_threshold, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        plt.title('Greek Sentiment Trend')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Value')
    
    # Plot 3: Regime transitions
    if 'Market_Regime' in daily_data.columns and date_col:
        plt.subplot(2, 2, 3)
        
        # Create a numeric mapping for regimes
        regimes = daily_data['Market_Regime'].unique()
        regime_to_num = {regime: i for i, regime in enumerate(sorted(regimes))}
        
        # Convert regimes to numeric values
        regime_numeric = daily_data['Market_Regime'].map(regime_to_num)
        
        # Plot regime transitions
        plt.plot(daily_data[date_col], regime_numeric, 'b-')
        
        # Set y-ticks to regime names
        plt.yticks(
            [regime_to_num[regime] for regime in sorted(regimes)],
            sorted(regimes)
        )
        
        plt.title('Greek Market Regime Transitions')
        plt.xlabel('Date')
        plt.ylabel('Market Regime')
    
    # Plot 4: Trading signals
    if 'Trading_Signal' in daily_data.columns and date_col:
        plt.subplot(2, 2, 4)
        
        # Count trading signals
        signal_counts = daily_data['Trading_Signal'].value_counts()
        
        # Create bar chart
        sns.barplot(x=signal_counts.index, y=signal_counts.values)
        
        plt.title('Greek Trading Signals')
        plt.xlabel('Signal')
        plt.ylabel('Count')
    
    # Add title to the figure
    plt.suptitle('Greek Market Regime Dashboard for Day Traders', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save dashboard
    dashboard_file = os.path.join(output_dir, f'greek_day_trader_dashboard_{timestamp}.png')
    plt.savefig(dashboard_file)
    plt.close()
    
    visualization_files.append(dashboard_file)
    logger.info(f"Greek day trader dashboard saved to {dashboard_file}")
    
    return visualization_files

def main():
    """
    Main function for daily Greek market regime analysis.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create day trader configuration
    config = create_day_trader_config(args)
    
    # Load and process data
    data = load_and_process_data(config, args)
    
    if data is None:
        logger.error("Failed to load and process data")
        return
    
    # Classify daily regimes
    daily_data = classify_daily_regimes(data, config)
    
    # Analyze intraday regimes if enabled
    intraday_data = None
    if args.intraday:
        intraday_data = analyze_intraday_regimes(daily_data, config)
    
    # Generate daily report
    report_file = generate_daily_report(daily_data, intraday_data, config, args)
    
    # Create visualizations
    visualization_files = create_visualization(daily_data, intraday_data, config, args)
    
    # Print summary
    logger.info("Greek daily market regime analysis completed")
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"Visualizations saved to: {args.output_dir}/visualizations")
    
    # Return results
    return {
        'report_file': report_file,
        'visualization_files': visualization_files,
        'daily_data': daily_data,
        'intraday_data': intraday_data
    }

if __name__ == "__main__":
    main()
