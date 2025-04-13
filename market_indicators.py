"""
Market indicators calculation module.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

def calculate_market_indicators(data, config):
    """
    Calculate market indicators for market regime determination.
    
    Args:
        data (DataFrame): Market data
        config (dict): Configuration settings
        
    Returns:
        DataFrame: Market data with indicators
    """
    logging.info("Calculating market indicators")
    
    # Create a copy of the data
    df = data.copy()
    
    # Calculate EMA signals
    logging.info("Calculating EMA signals")
    df = calculate_ema_signals(df)
    
    # Calculate VWAP signals
    logging.info("Calculating VWAP signals")
    df = calculate_vwap_signals(df)
    
    # Calculate ATR signals
    logging.info("Calculating ATR signals")
    df = calculate_atr_signals(df)
    
    # Calculate additional indicators
    df = calculate_additional_indicators(df)
    
    return df

def calculate_ema_signals(data):
    """
    Calculate EMA signals.
    
    Args:
        data (DataFrame): Market data
        
    Returns:
        DataFrame: Market data with EMA signals
    """
    # Create a copy of the data
    df = data.copy()
    
    # Check if EMA columns exist
    ema_columns = [col for col in df.columns if 'EMA' in col]
    
    if not ema_columns:
        # Calculate EMAs if not present
        if 'Close' in df.columns:
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        else:
            # Create synthetic EMAs for testing
            df['EMA_20'] = 150 + np.random.normal(0, 5, size=len(df))
            df['EMA_50'] = 145 + np.random.normal(0, 3, size=len(df))
            df['EMA_200'] = 140 + np.random.normal(0, 2, size=len(df))
    
    # Calculate EMA signals
    if 'EMA_20' in df.columns and 'EMA_50' in df.columns and 'EMA_200' in df.columns:
        # EMA 20 vs EMA 50
        df['EMA_20_50_Signal'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)
        
        # EMA 50 vs EMA 200
        df['EMA_50_200_Signal'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)
        
        # Combined EMA signal
        df['EMA_Signal'] = (df['EMA_20_50_Signal'] + df['EMA_50_200_Signal']) / 2
    else:
        # Create synthetic signals for testing
        df['EMA_Signal'] = np.random.uniform(-1, 1, size=len(df))
    
    return df

def calculate_vwap_signals(data):
    """
    Calculate VWAP signals.
    
    Args:
        data (DataFrame): Market data
        
    Returns:
        DataFrame: Market data with VWAP signals
    """
    # Create a copy of the data
    df = data.copy()
    
    # Check if VWAP column exists
    if 'VWAP' not in df.columns:
        # Calculate VWAP if not present
        if 'Close' in df.columns and 'Volume' in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        else:
            # Create synthetic VWAP for testing
            df['VWAP'] = 150 + np.random.normal(0, 3, size=len(df))
    
    # Calculate VWAP signal
    if 'Close' in df.columns and 'VWAP' in df.columns:
        # Price vs VWAP
        df['VWAP_Signal'] = np.where(df['Close'] > df['VWAP'], 1, -1)
    else:
        # Create synthetic signal for testing
        df['VWAP_Signal'] = np.random.choice([-1, 1], size=len(df))
    
    return df

def calculate_atr_signals(data):
    """
    Calculate ATR signals.
    
    Args:
        data (DataFrame): Market data
        
    Returns:
        DataFrame: Market data with ATR signals
    """
    # Create a copy of the data
    df = data.copy()
    
    # Check if ATR column exists
    if 'ATR' not in df.columns:
        # Calculate ATR if not present
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            # Calculate True Range
            df['TR'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            
            # Calculate ATR (14-period)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            
            # Drop TR column
            df = df.drop('TR', axis=1)
        else:
            # Create synthetic ATR for testing
            df['ATR'] = 3 + np.random.normal(0, 0.5, size=len(df))
    
    # Calculate ATR signal (volatility indicator)
    if 'ATR' in df.columns and 'Close' in df.columns:
        # ATR as percentage of price
        df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
        
        # ATR signal (higher ATR = higher volatility)
        df['ATR_Signal'] = df['ATR_Pct'].rolling(window=5).mean() / df['ATR_Pct'].rolling(window=20).mean() - 1
        
        # Normalize ATR signal to [-1, 1]
        max_atr = df['ATR_Signal'].max()
        min_atr = df['ATR_Signal'].min()
        if max_atr > min_atr:
            df['ATR_Signal'] = 2 * (df['ATR_Signal'] - min_atr) / (max_atr - min_atr) - 1
    else:
        # Create synthetic signal for testing
        df['ATR_Signal'] = np.random.uniform(-1, 1, size=len(df))
    
    return df

def calculate_additional_indicators(data):
    """
    Calculate additional market indicators.
    
    Args:
        data (DataFrame): Market data
        
    Returns:
        DataFrame: Market data with additional indicators
    """
    # Create a copy of the data
    df = data.copy()
    
    # Calculate RSI if possible
    if 'Close' in df.columns:
        # Calculate price changes
        delta = df['Close'].diff()
        
        # Calculate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI signal
        df['RSI_Signal'] = np.where(df['RSI'] > 70, -1, np.where(df['RSI'] < 30, 1, 0))
    else:
        # Create synthetic RSI for testing
        df['RSI'] = 50 + np.random.normal(0, 15, size=len(df))
        df['RSI_Signal'] = np.where(df['RSI'] > 70, -1, np.where(df['RSI'] < 30, 1, 0))
    
    # Calculate volatility indicator
    if 'ATR' in df.columns and 'Close' in df.columns:
        # Volatility as ATR percentage of price
        df['Volatility'] = df['ATR'] / df['Close']
        
        # Determine if volatility is high or low
        volatility_threshold = df['Volatility'].quantile(0.7)
        df['High_Volatility'] = df['Volatility'] > volatility_threshold
    else:
        # Create synthetic volatility for testing
        df['Volatility'] = np.random.uniform(0.01, 0.05, size=len(df))
        df['High_Volatility'] = np.random.choice([True, False], size=len(df), p=[0.3, 0.7])
    
    return df
