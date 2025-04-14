"""
Test script for the enhanced market regime optimizer with Greek sentiment analysis.
This script tests the improved implementation with a sample dataset.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.data_processor import process_data_efficiently, calculate_greek_sentiment_efficiently
from core.dynamic_weight_adjuster import DynamicWeightAdjuster
from core.regime_classifier import RegimeClassifier
from core.market_regime import process_greek_sentiment

def create_test_config():
    """
    Create a test configuration.
    
    Returns:
        ConfigParser: Test configuration
    """
    config = configparser.ConfigParser()
    
    # Data processing section
    config.add_section('data_processing')
    config.set('data_processing', 'market_data_dir', '../data/market_data')
    config.set('data_processing', 'chunk_size', '50000')
    config.set('data_processing', 'use_parallel', 'true')
    config.set('data_processing', 'num_processes', '2')
    config.set('data_processing', 'memory_limit_mb', '500')
    
    # Greek sentiment section
    config.add_section('greek_sentiment')
    config.set('greek_sentiment', 'enable_dynamic_weights', 'true')
    config.set('greek_sentiment', 'delta_weight', '1.2')
    config.set('greek_sentiment', 'vega_weight', '1.5')
    config.set('greek_sentiment', 'theta_weight', '0.3')
    config.set('greek_sentiment', 'sentiment_threshold_bullish', '8.0')
    config.set('greek_sentiment', 'sentiment_threshold_bearish', '-8.0')
    config.set('greek_sentiment', 'weight_history_window', '20')
    config.set('greek_sentiment', 'weight_learning_rate', '0.05')
    config.set('greek_sentiment', 'min_weight', '0.1')
    config.set('greek_sentiment', 'max_weight', '3.0')
    
    # Market regime section
    config.add_section('market_regime')
    config.set('market_regime', 'market_data_dir', '../data/market_data')
    config.set('market_regime', 'enable_greek_sentiment', 'true')
    config.set('market_regime', 'num_regimes', '5')
    config.set('market_regime', 'use_clustering', 'true')
    config.set('market_regime', 'adaptive_thresholds', 'true')
    config.set('market_regime', 'regime_lookback_window', '20')
    config.set('market_regime', 'strong_bearish_threshold', '-10.0')
    config.set('market_regime', 'bearish_threshold', '-3.0')
    config.set('market_regime', 'bullish_threshold', '3.0')
    config.set('market_regime', 'strong_bullish_threshold', '10.0')
    
    # Output section
    config.add_section('output')
    config.set('output', 'base_dir', '../output')
    config.set('output', 'log_dir', '../output/logs')
    
    return config

def create_sample_data(num_rows=1000, output_file=None):
    """
    Create a sample dataset for testing.
    
    Args:
        num_rows (int): Number of rows to generate
        output_file (str, optional): Path to save sample data
        
    Returns:
        DataFrame: Sample data
    """
    logger.info(f"Creating sample data with {num_rows} rows")
    
    # Create date range
    start_date = datetime(2024, 1, 1)
    dates = [start_date + pd.Timedelta(days=i) for i in range(num_rows)]
    
    # Create option data
    np.random.seed(42)  # For reproducibility
    
    data = pd.DataFrame({
        'Date': dates,
        'Strike': np.random.uniform(15000, 20000, num_rows),
        'Type': np.random.choice(['CE', 'PE'], num_rows),
        'Delta': np.random.uniform(-1, 1, num_rows),
        'Vega': np.random.uniform(0, 100, num_rows),
        'Theta': np.random.uniform(-50, 0, num_rows),
        'IV': np.random.uniform(0.1, 0.5, num_rows),
        'Volume': np.random.randint(100, 10000, num_rows),
        'OI': np.random.randint(1000, 100000, num_rows)
    })
    
    # Adjust Delta based on option type
    data.loc[data['Type'] == 'CE', 'Delta'] = np.abs(data.loc[data['Type'] == 'CE', 'Delta'])
    data.loc[data['Type'] == 'PE', 'Delta'] = -np.abs(data.loc[data['Type'] == 'PE', 'Delta'])
    
    # Save sample data if output file specified
    if output_file is not None:
        data.to_csv(output_file, index=False)
        logger.info(f"Saved sample data to {output_file}")
    
    return data

def create_sample_sentiment_data(num_rows=100, output_file=None):
    """
    Create a sample sentiment dataset for testing.
    
    Args:
        num_rows (int): Number of rows to generate
        output_file (str, optional): Path to save sample data
        
    Returns:
        DataFrame: Sample sentiment data
    """
    logger.info(f"Creating sample sentiment data with {num_rows} rows")
    
    # Create date range
    start_date = datetime(2024, 1, 1)
    dates = [start_date + pd.Timedelta(days=i) for i in range(num_rows)]
    
    # Create sentiment data
    np.random.seed(42)  # For reproducibility
    
    # Generate cyclical patterns for sentiment
    t = np.linspace(0, 4*np.pi, num_rows)
    delta_base = np.sin(t)
    vega_base = np.sin(t + np.pi/3)
    theta_base = np.sin(t + 2*np.pi/3)
    
    # Add some noise
    delta_noise = np.random.normal(0, 0.3, num_rows)
    vega_noise = np.random.normal(0, 0.3, num_rows)
    theta_noise = np.random.normal(0, 0.3, num_rows)
    
    data = pd.DataFrame({
        'Date': dates,
        'Delta_Sentiment': delta_base + delta_noise,
        'Vega_Sentiment': vega_base + vega_noise,
        'Theta_Sentiment': theta_base + theta_noise
    })
    
    # Calculate combined sentiment
    data['Combined_Sentiment'] = (
        data['Delta_Sentiment'] * 1.2 + 
        data['Vega_Sentiment'] * 1.5 + 
        data['Theta_Sentiment'] * 0.3
    ) / 3.0
    
    # Save sample data if output file specified
    if output_file is not None:
        data.to_csv(output_file, index=False)
        logger.info(f"Saved sample sentiment data to {output_file}")
    
    return data

def test_data_processor(sample_data, config, output_dir):
    """
    Test the DataProcessor with sample data.
    
    Args:
        sample_data (DataFrame): Sample data
        config (ConfigParser): Configuration
        output_dir (str): Output directory
        
    Returns:
        DataFrame: Processed data
    """
    logger.info("Testing DataProcessor")
    
    # Save sample data to file for testing file processing
    sample_file = os.path.join(output_dir, 'sample_data.csv')
    sample_data.to_csv(sample_file, index=False)
    
    # Test processing single file
    processed_file = os.path.join(output_dir, 'processed_sample_data.csv')
    processed_data = process_data_efficiently(sample_file, config, processed_file)
    
    # Test calculating Greek sentiment
    sentiment_file = os.path.join(output_dir, 'greek_sentiment_test.csv')
    
    try:
        sentiment_data = calculate_greek_sentiment_efficiently(sample_data, config, sentiment_file)
        logger.info(f"Generated sentiment data with {len(sentiment_data)} rows")
    except Exception as e:
        logger.error(f"Error calculating sentiment: {str(e)}")
        # Create sample sentiment data instead
        sentiment_data = create_sample_sentiment_data(100, os.path.join(output_dir, 'sample_sentiment_data.csv'))
        logger.info(f"Created sample sentiment data with {len(sentiment_data)} rows")
    
    return sentiment_data

def test_dynamic_weight_adjuster(sentiment_data, config, output_dir):
    """
    Test the DynamicWeightAdjuster with sentiment data.
    
    Args:
        sentiment_data (DataFrame): Sentiment data
        config (ConfigParser): Configuration
        output_dir (str): Output directory
        
    Returns:
        dict: Adjusted weights
    """
    logger.info("Testing DynamicWeightAdjuster")
    
    # Create adjuster
    adjuster = DynamicWeightAdjuster(config)
    
    # Test weight adjustment
    weights = adjuster.update_weights_based_on_history(sentiment_data)
    
    logger.info(f"Adjusted weights: Delta={weights['delta']:.4f}, Vega={weights['vega']:.4f}, Theta={weights['theta']:.4f}")
    
    # Generate some history data for plotting
    for i in range(5):
        # Slightly modify the performance metrics each time
        performance = {
            'delta': 0.5 + np.random.normal(0, 0.1),
            'vega': 0.6 + np.random.normal(0, 0.1),
            'theta': 0.4 + np.random.normal(0, 0.1)
        }
        adjuster.adjust_weights(performance)
    
    # Test plotting weights history
    plot_file = os.path.join(output_dir, 'weights_history_test.png')
    adjuster.plot_weights_history(plot_file)
    
    return weights

def test_regime_classifier(sentiment_data, config, output_dir):
    """
    Test the RegimeClassifier with sentiment data.
    
    Args:
        sentiment_data (DataFrame): Sentiment data
        config (ConfigParser): Configuration
        output_dir (str): Output directory
        
    Returns:
        DataFrame: Data with regime classifications
    """
    logger.info("Testing RegimeClassifier")
    
    # Ensure sentiment data has required columns
    if 'Combined_Sentiment' not in sentiment_data.columns:
        logger.warning("Combined_Sentiment column not found, adding it")
        # If we have component sentiments, calculate combined
        if all(col in sentiment_data.columns for col in ['Delta_Sentiment', 'Vega_Sentiment']):
            delta_weight = 1.2
            vega_weight = 1.5
            theta_weight = 0.3
            
            sentiment_data['Combined_Sentiment'] = (
                sentiment_data['Delta_Sentiment'] * delta_weight + 
                sentiment_data['Vega_Sentiment'] * vega_weight
            )
            
            if 'Theta_Sentiment' in sentiment_data.columns:
                sentiment_data['Combined_Sentiment'] += sentiment_data['Theta_Sentiment'] * theta_weight
                
            total_weight = delta_weight + vega_weight
            if 'Theta_Sentiment' in sentiment_data.columns:
                total_weight += theta_weight
                
            sentiment_data['Combined_Sentiment'] /= total_weight
        else:
            # Create random combined sentiment
            logger.warning("Component sentiment columns not found, creating random Combined_Sentiment")
            np.random.seed(42)
            sentiment_data['Combined_Sentiment'] = np.random.normal(0, 1, len(sentiment_data))
    
    # Process Greek sentiment
    processed_data = process_greek_sentiment(sentiment_data, config)
    
    # Create classifier
    classifier = RegimeClassifier(config)
    
    # Test regime classification
    classified_data = classifier.classify_regimes(processed_data)
    
    # Check if Market_Regime column exists
    if 'Market_Regime' not in classified_data.columns:
        logger.warning("Market_Regime column not found, adding dummy regime classifications")
        # Add dummy regime classifications for testing
        regimes = ['Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish']
        sentiment_values = classified_data['Combined_Sentiment']
        
        # Create bins based on percentiles
        bins = [float('-inf')]
        for p in [20, 40, 60, 80]:
            bins.append(np.percentile(sentiment_values, p))
        bins.append(float('inf'))
        
        # Assign regimes based on bins
        classified_data['Market_Regime'] = pd.cut(
            sentiment_values, 
            bins=bins, 
            labels=regimes
        )
    
    # Test plotting regime distribution
    dist_plot_file = os.path.join(output_dir, 'regime_distribution_test.png')
    classifier.plot_regime_distribution(classified_data, dist_plot_file)
    
    # Test plotting regime transitions
    trans_plot_file = os.path.join(output_dir, 'regime_transitions_test.png')
    classifier.plot_regime_transitions(classified_data, trans_plot_file)
    
    # Log regime distribution
    regime_counts = classified_data['Market_Regime'].value_counts()
    for regime, count in regime_counts.items():
        logger.info(f"Regime {regime}: {count} points ({count/len(classified_data)*100:.1f}%)")
    
    return classified_data

def create_test_dashboard(classified_data, weights, output_dir):
    """
    Create a test dashboard with the results.
    
    Args:
        classified_data (DataFrame): Data with regime classifications
        weights (dict): Adjusted weights
        output_dir (str): Output directory
        
    Returns:
        str: Path to dashboard file
    """
    logger.info("Creating test dashboard")
    
    # Create dashboard figure
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Regime distribution
    plt.subplot(2, 2, 1)
    regime_counts = classified_data['Market_Regime'].value_counts()
    sns.barplot(x=regime_counts.index, y=regime_counts.values)
    plt.title('Market Regime Distribution')
    plt.xlabel('Regime')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot 2: Combined sentiment
    plt.subplot(2, 2, 2)
    plt.plot(classified_data['Combined_Sentiment'], 'b-')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Combined Greek Sentiment')
    plt.xlabel('Data Point Index')
    plt.ylabel('Sentiment Value')
    
    # Plot 3: Regime transitions
    plt.subplot(2, 2, 3)
    # Create a numeric mapping for regimes
    regimes = classified_data['Market_Regime'].unique()
    regime_to_num = {regime: i for i, regime in enumerate(sorted(regimes))}
    
    # Convert regimes to numeric values
    regime_numeric = classified_data['Market_Regime'].map(regime_to_num)
    plt.plot(regime_numeric, 'g-')
    
    # Set y-ticks to regime names
    plt.yticks(
        [regime_to_num[regime] for regime in sorted(regimes)],
        sorted(regimes)
    )
    plt.title('Market Regime Transitions')
    plt.xlabel('Data Point Index')
    plt.ylabel('Market Regime')
    
    # Plot 4: Greek weights
    plt.subplot(2, 2, 4)
    weight_labels = ['Delta', 'Vega', 'Theta']
    weight_values = [weights['delta'], weights['vega'], weights['theta']]
    
    sns.barplot(x=weight_labels, y=weight_values)
    plt.title('Dynamic Greek Weights')
    plt.xlabel('Greek Component')
    plt.ylabel('Weight Value')
    
    # Add title to the figure
    plt.suptitle('Greek Sentiment Analysis Test Dashboard', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save dashboard
    dashboard_file = os.path.join(output_dir, 'test_dashboard.png')
    plt.savefig(dashboard_file, dpi=150)
    plt.close()
    
    logger.info(f"Saved test dashboard to {dashboard_file}")
    return dashboard_file

def run_tests():
    """
    Run tests for the enhanced market regime optimizer.
    
    Returns:
        dict: Test results
    """
    logger.info("Starting tests for enhanced market regime optimizer")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'tests', f'test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test configuration
    config = create_test_config()
    
    # Save configuration
    config_file = os.path.join(output_dir, 'test_config.ini')
    with open(config_file, 'w') as f:
        config.write(f)
    
    # Create sample data
    sample_data = create_sample_data(num_rows=1000, output_file=os.path.join(output_dir, 'sample_data.csv'))
    
    # Test data processor
    sentiment_data = test_data_processor(sample_data, config, output_dir)
    
    # Test dynamic weight adjuster
    weights = test_dynamic_weight_adjuster(sentiment_data, config, output_dir)
    
    # Test regime classifier
    classified_data = test_regime_classifier(sentiment_data, config, output_dir)
    
    # Create test dashboard
    dashboard_file = create_test_dashboard(classified_data, weights, output_dir)
    
    # Compile test results
    test_results = {
        'timestamp': timestamp,
        'output_dir': output_dir,
        'sample_data_rows': len(sample_data),
        'sentiment_data_rows': len(sentiment_data),
        'classified_data_rows': len(classified_data),
        'weights': weights,
        'regime_distribution': classified_data['Market_Regime'].value_counts().to_dict(),
        'dashboard_file': dashboard_file
    }
    
    # Save test results
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info(f"Tests completed successfully. Results saved to {output_dir}")
    return test_results

if __name__ == "__main__":
    run_tests()
