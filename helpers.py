"""
Helper functions for the zone optimization pipeline.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
from zipfile import ZipFile
import shutil
import glob

def setup_logging(log_file=None, log_level="INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to log file (optional)
        log_level (str): Logging level (default: INFO)
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup file logging
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a'
        )
    else:
        # Setup console logging
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to ensure exists
        
    Returns:
        str: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    return directory

def load_config(config_path):
    """
    Load configuration from INI file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration settings
    """
    import configparser
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Convert to dictionary for easier access
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            config_dict[section][key] = value
    
    return config_dict

def load_configuration(config_path):
    """
    Load configuration from INI file (alias for load_config).
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration settings
    """
    return load_config(config_path)

def find_files_by_pattern(directory, pattern):
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory (str): Directory to search
        pattern (str): File pattern to match
        
    Returns:
        list: List of matching file paths
    """
    if not os.path.exists(directory):
        logging.warning(f"Directory does not exist: {directory}")
        return []
    
    return glob.glob(os.path.join(directory, pattern))

def append_data(existing_data, new_data):
    """
    Append new data to existing data.
    
    Args:
        existing_data (DataFrame): Existing data (can be None)
        new_data (DataFrame): New data to append
        
    Returns:
        DataFrame: Combined data
    """
    if existing_data is None:
        return new_data
    
    return pd.concat([existing_data, new_data], ignore_index=True)

def get_timestamp():
    """
    Get current timestamp as string.
    
    Returns:
        str: Timestamp string
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_zip_file(zip_path, extract_dir):
    """
    Extract zip file to directory.
    
    Args:
        zip_path (str): Path to zip file
        extract_dir (str): Directory to extract to
        
    Returns:
        str: Path to extracted directory
    """
    # Create temp directory for extraction
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract files
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    return extract_dir

def find_nearest_date(target_date, date_list):
    """
    Find nearest date in a list of dates.
    
    Args:
        target_date (datetime.date): Target date
        date_list (list): List of dates
        
    Returns:
        datetime.date: Nearest date
    """
    if not date_list:
        return None
    
    # Convert to datetime.date if needed
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    
    # Convert all dates to datetime.date
    date_list = [pd.to_datetime(d).date() if not isinstance(d, datetime.date) else d for d in date_list]
    
    # Find nearest date
    nearest_date = min(date_list, key=lambda d: abs((d - target_date).days))
    
    return nearest_date

def find_nearest_time_index(time_list, target_time):
    """
    Find index of nearest time in a list of times.
    
    Args:
        time_list (list): List of times
        target_time (datetime.time): Target time
        
    Returns:
        int: Index of nearest time
    """
    if not time_list:
        return None
    
    # Convert to datetime.time if needed
    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time).time()
    
    # Convert all times to datetime.time
    time_list = [pd.to_datetime(t).time() if not isinstance(t, datetime.time) else t for t in time_list]
    
    # Convert times to seconds for comparison
    target_seconds = target_time.hour * 3600 + target_time.minute * 60 + target_time.second
    time_seconds = [t.hour * 3600 + t.minute * 60 + t.second for t in time_list]
    
    # Find nearest time
    nearest_idx = np.argmin(np.abs(np.array(time_seconds) - target_seconds))
    
    return nearest_idx

def find_nearest_strike_index(strikes, target_strike):
    """
    Find index of nearest strike in a list of strikes.
    
    Args:
        strikes (list): List of strike prices
        target_strike (float): Target strike price
        
    Returns:
        int: Index of nearest strike
    """
    if not strikes:
        return None
    
    # Find nearest strike
    nearest_idx = np.argmin(np.abs(np.array(strikes) - target_strike))
    
    return nearest_idx

def find_atm_strike(options_data):
    """
    Find at-the-money (ATM) strike price.
    
    Args:
        options_data (DataFrame): Options data
        
    Returns:
        float: ATM strike price
    """
    if 'Underlying_Price' in options_data.columns:
        # Use underlying price if available
        underlying_price = options_data['Underlying_Price'].iloc[0]
    else:
        # Estimate ATM strike as the strike with minimum absolute delta
        calls = options_data[options_data['OptionType'] == 'CALL']
        if len(calls) > 0 and 'Delta' in calls.columns:
            # Find strike with delta closest to 0.5
            atm_idx = np.argmin(np.abs(calls['Delta'] - 0.5))
            return calls.iloc[atm_idx]['Strike']
        else:
            # Use middle strike if delta not available
            return options_data['Strike'].median()
    
    # Find strike closest to underlying price
    atm_idx = np.argmin(np.abs(options_data['Strike'] - underlying_price))
    return options_data.iloc[atm_idx]['Strike']

def extract_dte_from_data(data, expiry_column):
    """
    Extract Days To Expiry (DTE) from data.
    
    Args:
        data (DataFrame): Data containing date and expiry date
        expiry_column (str): Name of expiry date column
        
    Returns:
        Series: DTE values
    """
    if expiry_column not in data.columns:
        logging.warning(f"Expiry column {expiry_column} not found in data")
        return pd.Series(0, index=data.index)
    
    # Convert dates to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data[expiry_column] = pd.to_datetime(data[expiry_column])
    
    # Calculate DTE
    dte = (data[expiry_column] - data['Date']).dt.days
    
    # Ensure DTE is not negative
    dte = dte.clip(lower=0)
    
    return dte

def save_to_csv(data, file_path):
    """
    Save data to CSV file.
    
    Args:
        data (DataFrame): Data to save
        file_path (str): Path to save file
        
    Returns:
        str: Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to CSV
    data.to_csv(file_path, index=False)
    
    return file_path

def save_to_excel(data, file_path):
    """
    Save data to Excel file.
    
    Args:
        data (DataFrame): Data to save
        file_path (str): Path to save file
        
    Returns:
        str: Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to Excel
    data.to_excel(file_path, index=False)
    
    return file_path
