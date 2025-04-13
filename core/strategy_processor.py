"""
Strategy data processing module.
"""

import pandas as pd
import numpy as np
import logging
import os
import glob
from datetime import datetime, timedelta

from utils.helpers import ensure_directory_exists

def process_strategy_data(config):
    """
    Second step: Process strategy data from input files.
    
    Args:
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Processed strategy data
    """
    logging.info("Starting strategy data processing")
    
    # Check if we should use TV files
    use_tv_files = config["input"].get("use_tv_files", "true").lower() == "true"
    
    # Check if we should use Python files
    use_python_files = config["input"].get("use_python_files", "true").lower() == "true"
    
    # Process TV zone files if enabled
    tv_data = None
    if use_tv_files:
        tv_data = process_tv_zone_files(config)
    
    # Process Python multi-zone files if enabled
    python_data = None
    if use_python_files:
        python_data = process_python_multi_zone_files(config)
    
    # Combine data if both sources are available
    if tv_data is not None and python_data is not None:
        # Check if we need to merge or use one source
        merge_strategy = config["input"].get("merge_strategy", "prefer_tv").lower()
        
        if merge_strategy == "combine":
            # Combine both sources
            combined_data = pd.concat([tv_data, python_data], ignore_index=True)
            
            # Remove duplicates if any
            combined_data = combined_data.drop_duplicates(subset=['Date', 'Time', 'Zone', 'Strategy'])
            
            logging.info(f"Combined {len(tv_data)} TV rows and {len(python_data)} Python rows into {len(combined_data)} rows")
            
            return combined_data
        elif merge_strategy == "prefer_python":
            # Prefer Python data
            logging.info(f"Using Python data with {len(python_data)} rows (prefer_python strategy)")
            return python_data
        else:
            # Default: prefer TV data
            logging.info(f"Using TV data with {len(tv_data)} rows (prefer_tv strategy)")
            return tv_data
    elif tv_data is not None:
        # Only TV data available
        logging.info(f"Using TV data with {len(tv_data)} rows (Python data not available)")
        return tv_data
    elif python_data is not None:
        # Only Python data available
        logging.info(f"Using Python data with {len(python_data)} rows (TV data not available)")
        return python_data
    else:
        # No data available, create synthetic data for testing
        logging.warning("No strategy data available, creating synthetic data for testing")
        return create_synthetic_strategy_data(config)

def process_tv_zone_files(config):
    """
    Process TradingView zone files.
    
    Args:
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Processed TV zone data
    """
    # Get TV zone files directory
    tv_zone_dir = config["input"].get("tv_zone_files_dir", "data/input/TV_Zone_Files")
    
    # Check if directory exists
    if not os.path.exists(tv_zone_dir):
        logging.warning(f"TV zone files directory {tv_zone_dir} does not exist")
        return None
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(tv_zone_dir, "*.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {tv_zone_dir}")
        return None
    
    # Process each file
    all_data = []
    
    for file in csv_files:
        try:
            # Load data
            data = pd.read_csv(file)
            
            # Extract zone name from filename
            zone_name = os.path.basename(file).replace(".csv", "").replace("_", " ")
            
            # Add zone column if not present
            if 'Zone' not in data.columns:
                data['Zone'] = zone_name
            
            # Add to all data
            all_data.append(data)
        except Exception as e:
            logging.error(f"Error processing TV zone file {file}: {str(e)}")
    
    if not all_data:
        logging.warning("No valid data found in TV zone files")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Ensure required columns exist
    required_columns = ['Date', 'Time', 'Zone', 'Strategy', 'PnL']
    
    for col in required_columns:
        if col not in combined_data.columns:
            if col == 'Strategy':
                # Use zone name as strategy if not present
                combined_data['Strategy'] = combined_data['Zone']
            elif col == 'PnL':
                # Use random PnL if not present
                combined_data['PnL'] = np.random.normal(0, 100, size=len(combined_data))
            else:
                logging.error(f"Required column {col} not found in TV zone files")
                return None
    
    # Add Day column if not present
    if 'Day' not in combined_data.columns:
        combined_data['Day'] = pd.to_datetime(combined_data['Date']).dt.day_name()
    
    # Add DTE column if not present
    if 'DTE' not in combined_data.columns:
        # Use random DTE values
        combined_data['DTE'] = np.random.randint(0, 7, size=len(combined_data))
    
    logging.info(f"Processed {len(combined_data)} rows from TV zone files")
    
    return combined_data

def process_python_multi_zone_files(config):
    """
    Process Python multi-zone files.
    
    Args:
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Processed Python multi-zone data
    """
    # Get Python multi-zone files directory
    python_zone_dir = config["input"].get("python_multi_zone_files_dir", "data/input/Python_Multi_Zone_Files")
    
    # Check if directory exists
    if not os.path.exists(python_zone_dir):
        logging.warning(f"Python multi-zone files directory {python_zone_dir} does not exist")
        return None
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(python_zone_dir, "*.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {python_zone_dir}")
        return None
    
    # Process each file
    all_data = []
    
    for file in csv_files:
        try:
            # Load data
            data = pd.read_csv(file)
            
            # Add to all data
            all_data.append(data)
        except Exception as e:
            logging.error(f"Error processing Python multi-zone file {file}: {str(e)}")
    
    if not all_data:
        logging.warning("No valid data found in Python multi-zone files")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Ensure required columns exist
    required_columns = ['Date', 'Time', 'Zone', 'Strategy', 'PnL']
    
    for col in required_columns:
        if col not in combined_data.columns:
            if col == 'Strategy':
                # Use zone name as strategy if not present
                combined_data['Strategy'] = combined_data['Zone']
            elif col == 'PnL':
                # Use random PnL if not present
                combined_data['PnL'] = np.random.normal(0, 100, size=len(combined_data))
            else:
                logging.error(f"Required column {col} not found in Python multi-zone files")
                return None
    
    # Add Day column if not present
    if 'Day' not in combined_data.columns:
        combined_data['Day'] = pd.to_datetime(combined_data['Date']).dt.day_name()
    
    # Add DTE column if not present
    if 'DTE' not in combined_data.columns:
        # Use random DTE values
        combined_data['DTE'] = np.random.randint(0, 7, size=len(combined_data))
    
    logging.info(f"Processed {len(combined_data)} rows from Python multi-zone files")
    
    return combined_data

def create_synthetic_strategy_data(config):
    """
    Create synthetic strategy data for testing.
    
    Args:
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Synthetic strategy data
    """
    logging.info("Creating synthetic strategy data for testing")
    
    # Create dates and times
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    times = pd.date_range(start='09:30:00', end='16:00:00', freq='30min').time
    
    # Create zones and strategies
    zones = ['Zone 1', 'Zone 2', 'Zone 3']
    strategies = [f'Strategy{i+1}' for i in range(len(zones))]
    
    # Create data rows
    data = []
    
    for date in dates:
        for time in times:
            for i, zone in enumerate(zones):
                # Add strategy data
                row = {
                    'Date': date.strftime('%Y-%m-%d'),
                    'Time': time.strftime('%H:%M:%S'),
                    'Zone': zone,
                    'Strategy': strategies[i],
                    'PnL': (i + 1) * 100 * (0.5 - (date.day % 3) * 0.3),
                    'DTE': date.day % 7
                }
                data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add Day column
    df['Day'] = pd.to_datetime(df['Date']).dt.day_name()
    
    logging.info(f"Created synthetic strategy data with {len(df)} rows")
    
    return df
