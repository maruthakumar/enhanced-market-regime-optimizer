"""
Greek Market Regime Module

This module provides functionality for processing Greek sentiment data and forming market regimes
based on Greek sentiment indicators (Delta, Vega, Theta).
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_greek_sentiment(data, config):
    """
    Process Greek sentiment data and determine market regimes.
    
    Args:
        data (DataFrame): Data with Greek sentiment components
        config (ConfigParser): Configuration parameters
        
    Returns:
        DataFrame: Data with processed Greek sentiment and market regimes
    """
    logger.info("Processing Greek sentiment data")
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Check if required columns exist
    required_columns = ['Delta_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment']
    missing_columns = [col for col in required_columns if col not in result.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns for Greek sentiment processing: {missing_columns}")
        
        # Add missing columns with random values for testing
        for col in missing_columns:
            result[col] = np.random.normal(0, 1, len(result))
    
    # Get dynamic weight adjustment setting
    enable_dynamic_weights = config.getboolean('greek_sentiment', 'enable_dynamic_weights', fallback=True)
    
    if enable_dynamic_weights:
        logger.info("Using dynamic weight adjustment based on historical performance")
        
        # Import here to avoid circular imports
        from greek_dynamic_weight_adjuster import GreekDynamicWeightAdjuster
        
        # Create weight adjuster
        weight_adjuster = GreekDynamicWeightAdjuster(config)
        
        # Evaluate component performance
        logger.info("Evaluating Greek sentiment component performance")
        performance = weight_adjuster.evaluate_component_performance(result)
        
        # Adjust weights based on performance
        logger.info("Adjusting weights based on component performance")
        weights = weight_adjuster.adjust_weights(result)
        
        # Get adjusted weights
        delta_weight = weights['delta_weight']
        vega_weight = weights['vega_weight']
        theta_weight = weights['theta_weight']
        
        logger.info(f"Adjusted weights: Delta={delta_weight:.4f}, Vega={vega_weight:.4f}, Theta={theta_weight:.4f}")
    else:
        # Use fixed weights from config
        delta_weight = config.getfloat('greek_sentiment', 'delta_weight', fallback=1.2)
        vega_weight = config.getfloat('greek_sentiment', 'vega_weight', fallback=1.5)
        theta_weight = config.getfloat('greek_sentiment', 'theta_weight', fallback=0.3)
        
        logger.info(f"Using fixed weights: Delta={delta_weight:.4f}, Vega={vega_weight:.4f}, Theta={theta_weight:.4f}")
    
    # Calculate total weight
    total_weight = delta_weight + vega_weight + theta_weight
    
    # Calculate combined sentiment
    result['Combined_Sentiment'] = (
        result['Delta_Sentiment'] * delta_weight +
        result['Vega_Sentiment'] * vega_weight +
        result['Theta_Sentiment'] * theta_weight
    ) / total_weight
    
    # Log dynamic weights
    logger.info(f"Dynamic weights: Delta={delta_weight:.4f}, Vega={vega_weight:.4f}, Theta={theta_weight:.4f}")
    
    # Determine sentiment labels
    logger.info("Determining sentiment labels")
    
    # Get threshold values
    bullish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bullish', fallback=8.0)
    bearish_threshold = config.getfloat('greek_sentiment', 'sentiment_threshold_bearish', fallback=-8.0)
    
    # Apply thresholds
    result['Sentiment_Label'] = 'Neutral'
    result.loc[result['Combined_Sentiment'] >= bullish_threshold, 'Sentiment_Label'] = 'Bullish'
    result.loc[result['Combined_Sentiment'] <= bearish_threshold, 'Sentiment_Label'] = 'Bearish'
    
    logger.info("Greek sentiment processing complete")
    
    return result

def classify_greek_market_regimes(data, config):
    """
    Classify market regimes based on Greek sentiment.
    
    Args:
        data (DataFrame): Data with processed Greek sentiment
        config (ConfigParser): Configuration parameters
        
    Returns:
        DataFrame: Data with market regime classifications
    """
    logger.info("Classifying market regimes based on Greek sentiment")
    
    # Import here to avoid circular imports
    from greek_regime_classifier import GreekRegimeClassifier
    
    # Create classifier
    classifier = GreekRegimeClassifier(config)
    
    # Classify regimes
    result = classifier.classify_regimes(data)
    
    return result

def generate_greek_trading_signals(data, config):
    """
    Generate trading signals based on Greek sentiment and market regimes.
    
    Args:
        data (DataFrame): Data with market regime classifications
        config (ConfigParser): Configuration parameters
        
    Returns:
        DataFrame: Data with trading signals
    """
    logger.info("Generating trading signals based on Greek sentiment")
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Check if required columns exist
    if 'Market_Regime' not in result.columns:
        logger.warning("Market_Regime column not found, cannot generate trading signals")
        return result
    
    # Generate trading signals based on market regime
    result['Trading_Signal'] = 'Hold'
    
    # Apply trading rules
    result.loc[result['Market_Regime'] == 'Strong Bullish', 'Trading_Signal'] = 'Strong Buy'
    result.loc[result['Market_Regime'] == 'Bullish', 'Trading_Signal'] = 'Buy'
    result.loc[result['Market_Regime'] == 'Bearish', 'Trading_Signal'] = 'Sell'
    result.loc[result['Market_Regime'] == 'Strong Bearish', 'Trading_Signal'] = 'Strong Sell'
    
    # Check for regime transitions
    if 'Regime_Shift' in result.columns and 'Transition_Type' in result.columns:
        # Override signals on regime transitions
        for i in range(1, len(result)):
            if result.iloc[i]['Regime_Shift']:
                transition = result.iloc[i]['Transition_Type']
                
                if transition == 'Bullish':
                    result.iloc[i, result.columns.get_loc('Trading_Signal')] = 'Buy'
                elif transition == 'Bearish':
                    result.iloc[i, result.columns.get_loc('Trading_Signal')] = 'Sell'
    
    return result
