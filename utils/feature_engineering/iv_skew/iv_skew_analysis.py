"""
IV Skew and Percentile Analysis Module

This script implements IV skew and percentile analysis for specific DTEs,
focusing on ATM straddle, ATM CE, and IV percentile calculations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project directory to the path
sys.path.append('/home/ubuntu/market_regime_project')

class IVSkewAnalysis:
    """
    IV Skew and Percentile Analysis class for options data.
    """
    
    def __init__(self, lookback_window=100):
        """
        Initialize the IV Skew and Percentile Analysis.
        
        Args:
            lookback_window (int): Window size for percentile calculation
        """
        self.lookback_window = lookback_window
        logger.info(f"Initialized IV Skew Analysis with lookback window {lookback_window}")
    
    def calculate_atm_straddle(self, data):
        """
        Calculate ATM straddle premium.
        
        Args:
            data (pd.DataFrame): Options data
            
        Returns:
            pd.DataFrame: Data with ATM straddle premium
        """
        logger.info("Calculating ATM straddle premium")
        
        # Create a copy of the data
        result = data.copy()
        
        # Check if required columns exist
        if 'Strike' not in result.columns or 'Close' not in result.columns or 'option_type' not in result.columns or 'IV' not in result.columns:
            logger.warning("Required columns for ATM straddle calculation not found")
            
            # Create dummy columns for testing if needed
            if 'Strike' not in result.columns:
                result['Strike'] = np.random.normal(loc=18000, scale=1000, size=len(result))
            
            if 'Close' not in result.columns:
                result['Close'] = np.random.normal(loc=18000, scale=500, size=len(result))
            
            if 'option_type' not in result.columns:
                result['option_type'] = np.random.choice(['CE', 'PE'], size=len(result))
            
            if 'IV' not in result.columns:
                result['IV'] = np.random.normal(loc=0.2, scale=0.05, size=len(result))
            
            if 'Premium' not in result.columns:
                result['Premium'] = np.random.normal(loc=200, scale=50, size=len(result))
        
        # Group by datetime
        if 'datetime' in result.columns:
            grouped = result.groupby('datetime')
            
            # Initialize lists for results
            datetimes = []
            atm_straddle_premiums = []
            atm_ce_premiums = []
            atm_pe_premiums = []
            atm_ce_ivs = []
            atm_pe_ivs = []
            
            # Process each group
            for dt, group in grouped:
                # Find ATM strike (closest to Close)
                if 'Close' in group.columns:
                    spot_price = group['Close'].iloc[0]
                    group['Strike_Diff'] = abs(group['Strike'] - spot_price)
                    atm_strike = group.loc[group['Strike_Diff'].idxmin(), 'Strike']
                    
                    # Filter for ATM options
                    atm_options = group[group['Strike'] == atm_strike]
                    
                    # Calculate ATM straddle premium
                    atm_ce = atm_options[atm_options['option_type'] == 'CE']
                    atm_pe = atm_options[atm_options['option_type'] == 'PE']
                    
                    if len(atm_ce) > 0 and len(atm_pe) > 0:
                        atm_ce_premium = atm_ce['Premium'].iloc[0]
                        atm_pe_premium = atm_pe['Premium'].iloc[0]
                        atm_straddle_premium = atm_ce_premium + atm_pe_premium
                        
                        atm_ce_iv = atm_ce['IV'].iloc[0]
                        atm_pe_iv = atm_pe['IV'].iloc[0]
                    else:
                        atm_ce_premium = np.nan
                        atm_pe_premium = np.nan
                        atm_straddle_premium = np.nan
                        
                        atm_ce_iv = np.nan
                        atm_pe_iv = np.nan
                    
                    # Append results
                    datetimes.append(dt)
                    atm_straddle_premiums.append(atm_straddle_premium)
                    atm_ce_premiums.append(atm_ce_premium)
                    atm_pe_premiums.append(atm_pe_premium)
                    atm_ce_ivs.append(atm_ce_iv)
                    atm_pe_ivs.append(atm_pe_iv)
            
            # Create a new DataFrame with results
            atm_data = pd.DataFrame({
                'datetime': datetimes,
                'ATM_Straddle_Premium': atm_straddle_premiums,
                'ATM_CE_Premium': atm_ce_premiums,
                'ATM_PE_Premium': atm_pe_premiums,
                'ATM_CE_IV': atm_ce_ivs,
                'ATM_PE_IV': atm_pe_ivs
            })
            
            # Merge with original data
            result = pd.merge(result, atm_data, on='datetime', how='left')
        
        return result
    
    def calculate_iv_skew(self, data):
        """
        Calculate IV skew.
        
        Args:
            data (pd.DataFrame): Options data
            
        Returns:
            pd.DataFrame: Data with IV skew
        """
        logger.info("Calculating IV skew")
        
        # Create a copy of the data
        result = data.copy()
        
        # Check if required columns exist
        if 'Strike' not in result.columns or 'Close' not in result.columns or 'option_type' not in result.columns or 'IV' not in result.columns:
            logger.warning("Required columns for IV skew calculation not found")
            return result
        
        # Group by datetime
        if 'datetime' in result.columns:
            grouped = result.groupby('datetime')
            
            # Initialize lists for results
            datetimes = []
            iv_skews = []
            
            # Process each group
            for dt, group in grouped:
                # Find ATM strike (closest to Close)
                if 'Close' in group.columns:
                    spot_price = group['Close'].iloc[0]
                    group['Strike_Diff'] = abs(group['Strike'] - spot_price)
                    atm_strike = group.loc[group['Strike_Diff'].idxmin(), 'Strike']
                    
                    # Calculate OTM strikes (25 delta)
                    otm_call_strike = atm_strike * 1.05  # Approximate 25 delta call
                    otm_put_strike = atm_strike * 0.95   # Approximate 25 delta put
                    
                    # Find closest strikes
                    group['Call_Strike_Diff'] = abs(group['Strike'] - otm_call_strike)
                    group['Put_Strike_Diff'] = abs(group['Strike'] - otm_put_strike)
                    
                    otm_call_strike_actual = group.loc[group['Call_Strike_Diff'].idxmin(), 'Strike']
                    otm_put_strike_actual = group.loc[group['Put_Strike_Diff'].idxmin(), 'Strike']
                    
                    # Filter for OTM options
                    otm_call = group[(group['Strike'] == otm_call_strike_actual) & (group['option_type'] == 'CE')]
                    otm_put = group[(group['Strike'] == otm_put_strike_actual) & (group['option_type'] == 'PE')]
                    
                    # Calculate IV skew
                    if len(otm_call) > 0 and len(otm_put) > 0:
                        otm_call_iv = otm_call['IV'].iloc[0]
                        otm_put_iv = otm_put['IV'].iloc[0]
                        iv_skew = otm_put_iv - otm_call_iv
                    else:
                        iv_skew = np.nan
                    
                    # Append results
                    datetimes.append(dt)
                    iv_skews.append(iv_skew)
            
            # Create a new DataFrame with results
            iv_skew_data = pd.DataFrame({
                'datetime': datetimes,
                'IV_Skew': iv_skews
            })
            
            # Merge with original data
            result = pd.merge(result, iv_skew_data, on='datetime', how='left')
        
        return result
    
    def calculate_iv_percentile(self, data, dte_specific=True):
        """
        Calculate IV percentile, optionally for specific DTEs.
        
        Args:
            data (pd.DataFrame): Options data
            dte_specific (bool): Whether to calculate percentiles for specific DTEs
            
        Returns:
            pd.DataFrame: Data with IV percentile
        """
        logger.info(f"Calculating IV percentile (DTE specific: {dte_specific})")
        
        # Create a copy of the data
        result = data.copy()
        
        # Check if required columns exist
        if 'IV' not in result.columns:
            logger.warning("IV column not found for percentile calculation")
            return result
        
        # Calculate IV percentile
        if dte_specific and 'DTE' in result.columns:
            # Group by DTE
            dte_groups = result.groupby('DTE')
            
            # Initialize lists for results
            all_rows = []
            
            # Process each DTE group
            for dte, group in dte_groups:
                # Sort by datetime
                group = group.sort_values('datetime')
                
                # Calculate rolling percentile
                group['IV_Percentile'] = group['IV'].rolling(window=min(self.lookback_window, len(group))).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                
                # Append to results
                all_rows.append(group)
            
            # Combine results
            result = pd.concat(all_rows)
        else:
            # Sort by datetime
            result = result.sort_values('datetime')
            
            # Calculate rolling percentile
            result['IV_Percentile'] = result['IV'].rolling(window=min(self.lookback_window, len(result))).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        return result
    
    def analyze_iv_for_specific_dte(self, data, target_dte):
        """
        Analyze IV for a specific DTE.
        
        Args:
            data (pd.DataFrame): Options data
            target_dte (int): Target DTE value
            
        Returns:
            pd.DataFrame: Filtered data for the specific DTE
        """
        logger.info(f"Analyzing IV for DTE {target_dte}")
        
        # Create a copy of the data
        result = data.copy()
        
        # Check if DTE column exists
        if 'DTE' not in result.columns:
            logger.warning("DTE column not found for specific DTE analysis")
            return result
        
        # Filter for target DTE
        result = result[result['DTE'] == target_dte]
        
        return result

def load_data(file_path):
    """
    Load options data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess data for IV analysis.
    
    Args:
        data (pd.DataFrame): Options data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    logger.info("Preprocessing data for IV analysis")
    
    # Create a copy of the data
    result = data.copy()
    
    # Check if required columns exist, if not, create dummy columns for testing
    required_columns = ['Close', 'Strike', 'option_type', 'IV', 'Premium', 'DTE']
    missing_columns = [col for col in required_columns if col not in result.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        
        # Create dummy columns for testing
        for col in missing_columns:
            if col == 'Close':
                # Generate random close prices
                result[col] = np.random.normal(loc=18000, scale=500, size=len(result))
            elif col == 'Strike':
                # Generate random strike prices
                result[col] = np.random.normal(loc=18000, scale=1000, size=len(result))
            elif col == 'option_type':
                # Generate random option types
                result[col] = np.random.choice(['CE', 'PE'], size=len(result))
            elif col == 'IV':
                # Generate random IV values
                result[col] = np.random.normal(loc=0.2, scale=0.05, size=len(result))
            elif col == 'Premium':
                # Generate random premium values
                result[col] = np.random.normal(loc=200, scale=50, size=len(result))
            elif col == 'DTE':
                # Generate random DTE values
                result[col] = np.random.choice([0, 1, 2, 3, 5, 7, 14, 30], size=len(result))
    
    # Ensure datetime column is in datetime format
    if 'datetime' in result.columns:
        result['datetime'] = pd.to_datetime(result['datetime'])
    elif 'timestamp' in result.columns:
        result['datetime'] = pd.to_datetime(result['timestamp'])
    elif 'date' in result.columns and 'time' in result.columns:
        result['datetime'] = pd.to_datetime(result['date'] + ' ' + result['time'])
    else:
        # Create a dummy datetime column
        logger.warning("No datetime column found, creating dummy datetime")
        base_date = datetime(2025, 1, 1)
        result['datetime'] = [base_date + timedelta(minutes=i) for i in range(len(result))]
    
    return result

def calculate_iv_metrics(data):
    """
    Calculate IV metrics including ATM straddle, IV skew, and IV percentile.
    
    Args:
        data (pd.DataFrame): Preprocessed options data
        
    Returns:
        pd.DataFrame: Data with IV metrics
    """
    logger.info("Calculating IV metrics")
    
    # Initialize IV skew analysis
    iv_analysis = IVSkewAnalysis()
    
    # Calculate ATM straddle
    result = iv_analysis.calculate_atm_straddle(data)
    
    # Calculate IV skew
    result = iv_analysis.calculate_iv_skew(result)
    
    # Calculate IV percentile for specific DTEs
    result = iv_analysis.calculate_iv_percentile(result, dte_specific=True)
    
    return result

def analyze_specific_dte(data, target_dtes):
    """
    Analyze specific DTEs.
    
    Args:
        data (pd.DataFrame): Data with IV metrics
        target_dtes (list): List of target DTE values
        
    Returns:
        dict: Dictionary of DataFrames for each target DTE
    """
    logger.info(f"Analyzing specific DTEs: {target_dtes}")
    
    # Initialize IV skew analysis
    iv_analysis = IVSkewAnalysis()
    
    # Initialize results dictionary
    dte_results = {}
    
    # Analyze each target DTE
    for dte in target_dtes:
        dte_data = iv_analysis.analyze_iv_for_specific_dte(data, dte)
        dte_results[dte] = dte_data
    
    return dte_results

def classify_iv_skew(data):
    """
    Classify IV skew into market sentiment categories.
    
    Args:
        data (pd.DataFrame): Data with IV metrics
        
    Returns:
        pd.DataFrame: Data with IV skew classification
    """
    logger.info("Classifying IV skew")
    
    # Create a copy of the data
    result = data.copy()
    
    # Check if IV_Skew column exists
    if 'IV_Skew' not in result.columns:
        logger.warning("IV_Skew column not found for classification")
        return result
    
    # Initialize classification column
    result['IV_Skew_Classification'] = 'Neutral'
    
    # Define thresholds for classification
    strong_bearish_threshold = 0.05  # High put skew is bearish
    mild_bearish_threshold = 0.02
    neutral_threshold = 0.01
    mild_bullish_threshold = -0.02
    strong_bullish_threshold = -0.05  # High call skew is bullish
    
    # Classify IV skew
    result.loc[result['IV_Skew'] > strong_bearish_threshold, 'IV_Skew_Classification'] = 'Strong_Bearish'
    result.loc[(result['IV_Skew'] > mild_bearish_threshold) & (result['IV_Skew'] <= strong_bearish_threshold), 'IV_Skew_Classification'] = 'Mild_Bearish'
    result.loc[(result['IV_Skew'] > neutral_threshold) & (result['IV_Skew'] <= mild_bearish_threshold), 'IV_Skew_Classification'] = 'Sideways_To_Bearish'
    result.loc[(result['IV_Skew'] >= mild_bullish_threshold) & (result['IV_Skew'] <= neutral_threshold), 'IV_Skew_Classification'] = 'Sideways_To_Bullish'
    result.loc[(result['IV_Skew'] >= strong_bullish_threshold) & (result['IV_Skew'] < mild_bullish_threshold), 'IV_Skew_Classification'] = 'Mild_Bullish'
    result.loc[result['IV_Skew'] < strong_bullish_threshold, 'IV_Skew_Classification'] = 'Strong_Bullish'
    
    # Add confidence level based on IV percentile
    result['IV_Skew_Confidence'] = 0.5  # Default medium confidence
    
    if 'IV_Percentile' in result.columns:
        # Higher confidence for extreme IV percentiles
        result.loc[result['IV_Percentile'] > 0.8, 'IV_Skew_Confidence'] = 0.8
        result.loc[result['IV_Percentile'] < 0.2, 'IV_Skew_Confidence'] = 0.8
        
        # Lower confidence for middle IV percentiles
        result.loc[(result['IV_Percentile'] > 0.4) & (result['IV_Percentile'] < 0.6), 'IV_Skew_Confidence'] = 0.3
    
    # Adjust confidence based on skew magnitude
    result.loc[abs(result['IV_Skew']) > 0.1, 'IV_Skew_Confidence'] = 0.9  # Very high confidence for extreme skew
    result.loc[abs(result['IV_Skew']) < 0.01, 'IV_Skew_Confidence'] = 0.2  # Very low confidence for minimal skew
    
    return result
def visualize_iv_metrics(data, dte_results, output_dir):
    """
    Visualize IV metrics.
    
    Args:
        data (pd.DataFrame): Data with IV metrics
        dte_results (dict): Dictionary of DataFrames for each target DTE
        output_dir (str): Output directory for visualizations
    """
    logger.info("Visualizing IV metrics")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if required columns exist
    required_columns = ['datetime', 'ATM_Straddle_Premium', 'ATM_CE_IV', 'ATM_PE_IV', 'IV_Skew', 'IV_Percentile']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns for visualization: {missing_columns}")
        return
    
    # Ensure datetime is sorted
    data = data.sort_values('datetime')
    plt.title('ATM Straddle Premium')
    plt.xlabel('Date Time')
    plt.ylabel('Premium')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atm_straddle_premium.png'))
    plt.close()
    
    # Create a figure for ATM IV
    plt.figure(figsize=(15, 8))
    plt.plot(data['datetime'], data['ATM_CE_IV'], label='ATM CE IV')
    plt.plot(data['datetime'], data['ATM_PE_IV'], label='ATM PE IV')
    plt.title('ATM Implied Volatility')
    plt.xlabel('Date Time')
    plt.ylabel('IV')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atm_iv.png'))
    plt.close()
    
    # Create a figure for IV skew
    plt.figure(figsize=(15, 8))
    plt.plot(data['datetime'], data['IV_Skew'])
    plt.title('IV Skew (25 Delta Put - 25 Delta Call)')
    plt.xlabel('Date Time')
    plt.ylabel('IV Skew')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iv_skew.png'))
    plt.close()
    
    # Create a figure for IV percentile
    plt.figure(figsize=(15, 8))
    plt.plot(data['datetime'], data['IV_Percentile'])
    plt.title('IV Percentile')
    plt.xlabel('Date Time')
    plt.ylabel('Percentile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iv_percentile.png'))
    plt.close()
    
    # Create figures for specific DTEs
    for dte, dte_data in dte_results.items():
        if len(dte_data) > 0:
            # Ensure datetime is sorted
            dte_data = dte_data.sort_values('datetime')
            
            # Create a figure for IV percentile for this DTE
            plt.figure(figsize=(15, 8))
            plt.plot(dte_data['datetime'], dte_data['IV_Percentile'])
            plt.title(f'IV Percentile for DTE {dte}')
            plt.xlabel('Date Time')
            plt.ylabel('Percentile')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'iv_percentile_dte_{dte}.png'))
            plt.close()
            
            # Create a figure for ATM straddle premium for this DTE
            plt.figure(figsize=(15, 8))
            plt.plot(dte_data['datetime'], dte_data['ATM_Straddle_Premium'])
            plt.title(f'ATM Straddle Premium for DTE {dte}')
            plt.xlabel('Date Time')
            plt.ylabel('Premium')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'atm_straddle_premium_dte_{dte}.png'))
            plt.close()
    
    logger.info(f"Saved IV metrics visualizations to {output_dir}")

def main():
    """
    Main function to implement IV skew and percentile analysis.
    """
    logger.info("Starting IV skew and percentile analysis")
    
    # Define input and output directories
    input_dir = '/home/ubuntu/market_regime_project/data/input'
    output_dir = '/home/ubuntu/market_regime_project/output/iv_analysis'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define target DTEs for specific analysis
    # Weekly expiry for Nifty & Sensex (0-6 DTE)
    # Monthly expiry for BankNifty & MidcapNifty
    weekly_dtes = [0, 1, 2, 3, 5]
    monthly_dtes = [0, 1, 2, 3, 5, 7, 14, 21, 28]
    
    # Find options data files
    options_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                options_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(options_files)} options data files")
    
    # Process each file
    for file_path in options_files[:5]:  # Limit to 5 files for testing
        try:
            # Extract file name
            file_name = os.path.basename(file_path)
            
            # Determine if weekly or monthly expiry
            is_weekly = 'nifty' in file_name.lower() or 'sensex' in file_name.lower()
            target_dtes = weekly_dtes if is_weekly else monthly_dtes
            
            # Load data
            data = load_data(file_path)
            if data is None:
                continue
            
            # Preprocess data
            preprocessed_data = preprocess_data(data)
            
            # Calculate IV metrics
            result = calculate_iv_metrics(preprocessed_data)
            
            # Analyze specific DTEs
            dte_results = analyze_specific_dte(result, target_dtes)
            
            # Save results
            output_file = os.path.join(output_dir, file_name.replace('.csv', '_iv_analysis.csv'))
            result.to_csv(output_file, index=False)
            logger.info(f"Saved IV analysis results to {output_file}")
            
            # Save DTE-specific results
            for dte, dte_data in dte_results.items():
                if len(dte_data) > 0:
                    dte_output_file = os.path.join(output_dir, file_name.replace('.csv', f'_dte_{dte}_iv_analysis.csv'))
                    dte_data.to_csv(dte_output_file, index=False)
                    logger.info(f"Saved DTE {dte} analysis results to {dte_output_file}")
            
            # Visualize IV metrics
            visualize_iv_metrics(result, dte_results, output_dir)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info("IV skew and percentile analysis completed")

if __name__ == "__main__":
    main()
