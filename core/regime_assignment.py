"""
Market regime assignment module.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def assign_market_regimes(strategy_data, market_regimes, config):
    """
    Third step: Assign market regimes to strategy data.
    
    Args:
        strategy_data (DataFrame): Strategy data
        market_regimes (DataFrame): Market regimes data
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Strategy data with assigned market regimes
    """
    logging.info("Starting market regime assignment")
    
    # Create a copy of the strategy data
    data = strategy_data.copy()
    
    # Check if market regimes data is available
    if market_regimes is None or len(market_regimes) == 0:
        logging.warning("No market regimes data available, using default regimes")
        return assign_default_regimes(data, config)
    
    # Ensure Date and Time columns are in the correct format
    data = standardize_datetime_format(data)
    market_regimes = standardize_datetime_format(market_regimes)
    
    # Assign market regimes based on nearest timestamp
    logging.info("Assigning market regimes using nearest timestamp")
    data = assign_regimes_by_timestamp(data, market_regimes, config)
    
    # Fill missing market regimes
    data = fill_missing_regimes(data, config)
    
    logging.info(f"Market regime assignment completed for {len(data)} rows")
    
    return data

def standardize_datetime_format(data):
    """
    Standardize Date and Time columns to ensure consistent format.
    
    Args:
        data (DataFrame): Data with Date and Time columns
        
    Returns:
        DataFrame: Data with standardized Date and Time columns
    """
    # Create a copy of the data
    df = data.copy()
    
    # Standardize Date column
    if 'Date' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                logging.warning(f"Error converting Date to datetime: {str(e)}")
        
        # Convert to string format YYYY-MM-DD
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Standardize Time column
    if 'Time' in df.columns:
        # Check if Time is already a time-like object by checking for a sample value
        is_time_object = False
        if len(df) > 0:
            sample_time = df['Time'].iloc[0]
            is_time_object = hasattr(sample_time, 'hour') and hasattr(sample_time, 'minute')
        
        if is_time_object:
            # Convert to string format HH:MM:SS
            df['Time'] = df['Time'].apply(lambda x: x.strftime('%H:%M:%S') if x is not None else None)
        else:
            # Try to convert to standard time format
            try:
                # Handle different time formats
                time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
                
                # Try each format
                for fmt in time_formats:
                    try:
                        # Convert to datetime first to handle different formats
                        temp_datetime = pd.to_datetime(df['Time'], format=fmt, errors='coerce')
                        
                        # Convert to string format HH:MM:SS
                        df['Time'] = temp_datetime.dt.strftime('%H:%M:%S')
                        
                        # Break if successful
                        if df['Time'].notna().all():
                            break
                    except Exception:
                        continue
            except Exception as e:
                logging.warning(f"Error standardizing Time format: {str(e)}")
    
    return df

def assign_regimes_by_timestamp(strategy_data, market_regimes, config):
    """
    Assign market regimes to strategy data based on nearest timestamp.
    
    Args:
        strategy_data (DataFrame): Strategy data
        market_regimes (DataFrame): Market regimes data
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Strategy data with assigned market regimes
    """
    # Create a copy of the strategy data
    data = strategy_data.copy()
    
    # Create datetime column for matching
    data['datetime'] = data['Date'] + ' ' + data['Time']
    market_regimes['datetime'] = market_regimes['Date'] + ' ' + market_regimes['Time']
    
    # Convert to datetime
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    market_regimes['datetime'] = pd.to_datetime(market_regimes['datetime'], errors='coerce')
    
    # Create a dictionary of market regimes by datetime
    regime_dict = {}
    for _, row in market_regimes.iterrows():
        if pd.notna(row['datetime']):
            regime_dict[row['datetime']] = row['Market regime']
    
    # Assign market regimes based on nearest timestamp
    data['Market regime'] = data['datetime'].apply(
        lambda x: find_nearest_regime(x, regime_dict) if pd.notna(x) else None
    )
    
    # Drop datetime column
    data = data.drop('datetime', axis=1)
    
    return data

def find_nearest_regime(timestamp, regime_dict):
    """
    Find the nearest market regime for a given timestamp.
    
    Args:
        timestamp (datetime): Timestamp to find regime for
        regime_dict (dict): Dictionary of market regimes by datetime
        
    Returns:
        str: Market regime
    """
    if not regime_dict:
        return None
    
    # Get all timestamps
    timestamps = list(regime_dict.keys())
    
    # Find nearest timestamp
    nearest = min(timestamps, key=lambda x: abs(x - timestamp))
    
    # Get regime for nearest timestamp
    return regime_dict[nearest]

def assign_default_regimes(data, config):
    """
    Assign default market regimes to strategy data.
    
    Args:
        data (DataFrame): Strategy data
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Strategy data with assigned default market regimes
    """
    # Create a copy of the data
    df = data.copy()
    
    # Create synthetic market regimes
    regimes = [
        'high_voltatile_strong_bullish',
        'high_voltatile_mild_bullish',
        'high_voltatile_sideways_neutral',
        'high_voltatile_mild_bearish',
        'high_voltatile_strong_bearish',
        'Low_volatole_strong_bullish',
        'Low_volatole_mild_bullish',
        'Low_volatole_sideways_bearish',
        'Low_volatole_mild_bearish',
        'Low_volatole_strong_bearish'
    ]
    
    # Assign random regimes
    df['Market regime'] = np.random.choice(regimes, size=len(df))
    
    logging.warning(f"Assigned default market regimes to {len(df)} rows")
    
    return df

def fill_missing_regimes(data, config):
    """
    Fill missing market regimes.
    
    Args:
        data (DataFrame): Strategy data with assigned market regimes
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Strategy data with filled market regimes
    """
    # Create a copy of the data
    df = data.copy()
    
    # Check if there are missing regimes
    missing_count = df['Market regime'].isna().sum()
    
    if missing_count > 0:
        logging.warning(f"Found {missing_count} rows with missing market regimes")
        
        # Fill missing regimes with default value
        df['Market regime'] = df['Market regime'].fillna('high_voltatile_sideways_neutral')
        
        logging.info(f"Filled {missing_count} missing market regimes")
    
    return df
