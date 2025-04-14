"""
Modified market regime module for the enhanced market regime optimizer.
This module provides functions to form market regimes based on various indicators,
including Greek sentiment analysis with dynamic weight adjustment.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.regime_assignment import assign_market_regimes
from core.dynamic_weight_adjuster import get_dynamic_weights

logger = logging.getLogger(__name__)

def form_market_regimes(config):
    """
    Form market regimes based on various indicators including Greek sentiment.
    
    Args:
        config (ConfigParser): Configuration parameters
        
    Returns:
        DataFrame: Market regime data
    """
    logger.info("Forming market regimes with Greek sentiment analysis")
    
    # Use config values or defaults
    if isinstance(config, dict):
        data_dir = config.get('market_data_dir') if 'market_data_dir' in config else '../data/market_data'
        base_dir = config.get('base_dir') if 'base_dir' in config else '../output'
        greek_sentiment_enabled = config.get('enable_greek_sentiment', False)
        dynamic_weights_enabled = config.get('enable_dynamic_weights', True)
    else:
        data_dir = config.get('market_regime', 'market_data_dir')
        base_dir = config.get('output', 'base_dir')
        greek_sentiment_enabled = config.getboolean('market_regime', 'enable_greek_sentiment')
        dynamic_weights_enabled = config.getboolean('greek_sentiment', 'enable_dynamic_weights', fallback=True)
    
    output_dir = os.path.join(base_dir, 'market_regime')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load market data
    try:
        # Try to load integrated data first
        if isinstance(config, dict):
            integrated_data_file = config.get('integrated_data_file') if 'integrated_data_file' in config else os.path.join(data_dir, 'integrated_data.csv')
        else:
            integrated_data_file = config.get('market_regime', 'integrated_data_file', fallback=os.path.join(data_dir, 'integrated_data.csv'))
        
        if os.path.exists(integrated_data_file):
            logger.info(f"Loading integrated data from {integrated_data_file}")
            market_data = pd.read_csv(integrated_data_file)
        else:
            # Try simple integrated data
            simple_data_file = integrated_data_file.replace('.csv', '_simple.csv')
            if os.path.exists(simple_data_file):
                logger.info(f"Loading simple integrated data from {simple_data_file}")
                market_data = pd.read_csv(simple_data_file)
            else:
                # Fallback to Greek sentiment data
                if isinstance(config, dict):
                    greek_sentiment_file = config.get('greek_sentiment_file') if 'greek_sentiment_file' in config else os.path.join(data_dir, 'greek_sentiment.csv')
                else:
                    greek_sentiment_file = config.get('market_regime', 'greek_sentiment_file', fallback=os.path.join(data_dir, 'greek_sentiment.csv'))
                
                if os.path.exists(greek_sentiment_file):
                    logger.info(f"Loading Greek sentiment data from {greek_sentiment_file}")
                    market_data = pd.read_csv(greek_sentiment_file)
                else:
                    logger.error("No valid data files found")
                    return None
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        return None
    
    logger.info(f"Loaded market data with {len(market_data)} rows")
    
    # Process Greek sentiment if enabled
    if greek_sentiment_enabled:
        # Load price data for performance evaluation if available
        price_data = None
        try:
            if isinstance(config, dict):
                price_data_file = config.get('price_data_file') if 'price_data_file' in config else os.path.join(data_dir, 'price_data.csv')
            else:
                price_data_file = config.get('market_regime', 'price_data_file', fallback=os.path.join(data_dir, 'price_data.csv'))
            
            if os.path.exists(price_data_file):
                logger.info(f"Loading price data from {price_data_file}")
                price_data = pd.read_csv(price_data_file)
        except Exception as e:
            logger.warning(f"Error loading price data: {str(e)}")
        
        market_data = process_greek_sentiment(market_data, config, price_data, dynamic_weights_enabled)
    
    # Assign market regimes
    regime_data = assign_market_regimes(market_data, config)
    
    # Save regime data
    output_file = os.path.join(output_dir, 'market_regimes.csv')
    regime_data.to_csv(output_file, index=False)
    logger.info(f"Saved market regime data to {output_file}")
    
    return regime_data

def process_greek_sentiment(data, config, price_data=None, dynamic_weights_enabled=True):
    """
    Process Greek sentiment data to enhance market regime detection.
    
    Args:
        data (DataFrame): Market data with Greek sentiment
        config (ConfigParser): Configuration parameters
        price_data (DataFrame, optional): Price data for performance evaluation
        dynamic_weights_enabled (bool): Whether to use dynamic weight adjustment
        
    Returns:
        DataFrame: Enhanced market data with processed sentiment
    """
    logger.info("Processing Greek sentiment data")
    
    # Get weights - either dynamic or from config
    if dynamic_weights_enabled:
        logger.info("Using dynamic weight adjustment based on historical performance")
        weights = get_dynamic_weights(data, config, price_data)
        vega_weight = weights['vega']
        delta_weight = weights['delta']
        theta_weight = weights['theta']
        
        # Log the dynamically determined weights
        logger.info(f"Dynamic weights: Delta={delta_weight:.4f}, Vega={vega_weight:.4f}, Theta={theta_weight:.4f}")
    else:
        # Get configuration parameters
        if isinstance(config, dict):
            vega_weight = config.get('vega_weight', 1.0)
            delta_weight = config.get('delta_weight', 1.0)
            theta_weight = config.get('theta_weight', 0.5)
        else:
            vega_weight = config.getfloat('greek_sentiment', 'vega_weight', fallback=1.0)
            delta_weight = config.getfloat('greek_sentiment', 'delta_weight', fallback=1.0)
            theta_weight = config.getfloat('greek_sentiment', 'theta_weight', fallback=0.5)
        
        logger.info(f"Using fixed weights: Delta={delta_weight:.4f}, Vega={vega_weight:.4f}, Theta={theta_weight:.4f}")
    
    # Check if we need to calculate combined sentiment
    if 'Combined_Sentiment' not in data.columns:
        # Check if we have the component sentiments
        if all(col in data.columns for col in ['Vega_Sentiment', 'Delta_Sentiment']):
            logger.info("Calculating combined sentiment from components")
            
            # Calculate weighted combined sentiment
            data['Combined_Sentiment'] = (
                data['Vega_Sentiment'] * vega_weight + 
                data['Delta_Sentiment'] * delta_weight
            )
            
            # Add theta component if available
            if 'Theta_Sentiment' in data.columns:
                data['Combined_Sentiment'] += data['Theta_Sentiment'] * theta_weight
                
            # Normalize by sum of weights
            total_weight = vega_weight + delta_weight
            if 'Theta_Sentiment' in data.columns:
                total_weight += theta_weight
                
            data['Combined_Sentiment'] /= total_weight
    
    # Determine sentiment labels if not already present
    if 'Sentiment' not in data.columns and 'Combined_Sentiment' in data.columns:
        logger.info("Determining sentiment labels")
        
        # Get thresholds from config
        if isinstance(config, dict):
            bullish_threshold = config.get('sentiment_threshold_bullish', 10.0)
            bearish_threshold = config.get('sentiment_threshold_bearish', -10.0)
        else:
            bullish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bullish', fallback=10.0)
            bearish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bearish', fallback=-10.0)
        
        # Apply thresholds
        data['Sentiment'] = 'Neutral'
        data.loc[data['Combined_Sentiment'] > bullish_threshold, 'Sentiment'] = 'Bullish'
        data.loc[data['Combined_Sentiment'] < bearish_threshold, 'Sentiment'] = 'Bearish'
    
    # Add sentiment strength (0-100 scale) for visualization
    if 'Combined_Sentiment' in data.columns and 'Sentiment_Strength' not in data.columns:
        # Calculate min/max for normalization
        sentiment_min = data['Combined_Sentiment'].min()
        sentiment_max = data['Combined_Sentiment'].max()
        
        # Normalize to 0-100 scale
        if sentiment_max > sentiment_min:
            data['Sentiment_Strength'] = (data['Combined_Sentiment'] - sentiment_min) / (sentiment_max - sentiment_min) * 100
        else:
            data['Sentiment_Strength'] = 50  # Default to neutral if no variation
    
    # Store the weights used for reference
    data['Delta_Weight'] = delta_weight
    data['Vega_Weight'] = vega_weight
    data['Theta_Weight'] = theta_weight
    
    logger.info("Greek sentiment processing complete")
    return data
