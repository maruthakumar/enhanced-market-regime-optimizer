"""
Greek Test Implementation Script

This script tests the implementation of Greek sentiment analysis for market regime optimization.
It generates sample data, processes it through the system, and creates visualizations to demonstrate
the functionality.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Greek-specific core modules
from greek_data_processor import process_data_efficiently, calculate_greek_sentiment_efficiently
from greek_dynamic_weight_adjuster import GreekDynamicWeightAdjuster
from greek_regime_classifier import GreekRegimeClassifier
from greek_market_regime import process_greek_sentiment, classify_greek_market_regimes

def create_test_config():
    """
    Create configuration for testing.
    
    Returns:
        ConfigParser: Configuration for testing
    """
    config = configparser.ConfigParser()
    
    # Data processing section
    config.add_section('data_processing')
    config.set('data_processing', 'market_data_dir', '../data/market_data')
    config.set('data_processing', 'chunk_size', '10000')
    config.set('data_processing', 'use_parallel', 'false')
    config.set('data_processing', 'num_processes', '2')
    config.set('data_processing', 'memory_limit_mb', '1000')
    
    # Greek sentiment section
    config.add_section('greek_sentiment')
    config.set('greek_sentiment', 'enable_dynamic_weights', 'true')
    config.set('greek_sentiment', 'delta_weight', '1.2')
    config.set('greek_sentiment', 'vega_weight', '1.5')
    config.set('greek_sentiment', 'theta_weight', '0.3')
    config.set('greek_sentiment', 'sentiment_threshold_bullish', '5.0')
    config.set('greek_sentiment', 'sentiment_threshold_bearish', '-5.0')
    config.set('greek_sentiment', 'weight_history_window', '10')
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
    config.set('market_regime', 'regime_lookback_window', '10')
    config.set('market_regime', 'strong_bearish_threshold', '-8.0')
    config.set('market_regime', 'bearish_threshold', '-2.5')
    config.set('market_regime', 'bullish_threshold', '2.5')
    config.set('market_regime', 'strong_bullish_threshold', '8.0')
    
    # Output section
    config.add_section('output')
    config.set('output', 'base_dir', '../output')
    
    return config

def generate_sample_data(num_samples=1000):
    """
    Generate sample data for testing.
    
    Args:
        num_samples (int, optional): Number of samples to generate
        
    Returns:
        DataFrame: Sample data
    """
    logger.info(f"Generating {num_samples} sample data points")
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)]
    
    # Generate option data
    data = {
        'Date': dates,
        'Strike': np.random.uniform(90, 110, num_samples),
        'Expiry': [start_date + timedelta(days=np.random.randint(1, 30)) for _ in range(num_samples)],
        'Price': np.random.uniform(1, 10, num_samples),
        'Delta': np.random.uniform(-1, 1, num_samples),
        'Gamma': np.random.uniform(0, 0.1, num_samples),
        'Vega': np.random.uniform(0, 0.5, num_samples),
        'Theta': np.random.uniform(-0.1, 0, num_samples),
        'Rho': np.random.uniform(-0.1, 0.1, num_samples),
        'IV': np.random.uniform(0.1, 0.5, num_samples),
        'Volume': np.random.randint(1, 1000, num_samples),
        'Open_Interest': np.random.randint(10, 5000, num_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add market return (for testing dynamic weight adjustment)
    df['Market_Return'] = np.random.normal(0, 0.01, num_samples)
    
    # Add some trends and patterns
    # Trend in Delta
    trend = np.linspace(-0.5, 0.5, num_samples)
    df['Delta'] = df['Delta'] + trend
    
    # Cyclical pattern in Vega
    cycle = np.sin(np.linspace(0, 6*np.pi, num_samples))
    df['Vega'] = df['Vega'] + cycle * 0.2
    
    # Regime shifts in Theta
    regime_shifts = np.zeros(num_samples)
    shift_points = np.random.choice(range(100, num_samples-100), 5, replace=False)
    for shift in shift_points:
        regime_shifts[shift:shift+100] = np.random.uniform(-0.05, 0.05)
    df['Theta'] = df['Theta'] + regime_shifts
    
    return df

def save_test_config(config, output_dir):
    """
    Save test configuration to file.
    
    Args:
        config (ConfigParser): Configuration
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved configuration file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(output_dir, 'test_config.ini')
    with open(config_file, 'w') as f:
        config.write(f)
    
    logger.info(f"Saved test configuration to {config_file}")
    
    return config_file

def test_greek_dynamic_weight_adjuster(data, config, output_dir):
    """
    Test the GreekDynamicWeightAdjuster.
    
    Args:
        data (DataFrame): Sample data
        config (ConfigParser): Configuration
        output_dir (str): Output directory
    """
    logger.info("Testing GreekDynamicWeightAdjuster")
    
    # Create weight adjuster
    weight_adjuster = GreekDynamicWeightAdjuster(config)
    
    # Evaluate component performance
    performance = weight_adjuster.evaluate_component_performance(data)
    
    # Adjust weights multiple times to see adaptation
    weights_history = []
    for i in range(10):
        weights = weight_adjuster.adjust_weights(data)
        weights_history.append(weights)
        logger.info(f"Iteration {i+1}: Delta={weights['delta_weight']:.4f}, Vega={weights['vega_weight']:.4f}, Theta={weights['theta_weight']:.4f}")
    
    # Plot weight history
    plt.figure(figsize=(12, 6))
    
    # Extract weights
    delta_weights = [w['delta_weight'] for w in weights_history]
    vega_weights = [w['vega_weight'] for w in weights_history]
    theta_weights = [w['theta_weight'] for w in weights_history]
    
    # Plot weights
    plt.plot(range(1, len(weights_history)+1), delta_weights, 'b-', label='Delta Weight')
    plt.plot(range(1, len(weights_history)+1), vega_weights, 'g-', label='Vega Weight')
    plt.plot(range(1, len(weights_history)+1), theta_weights, 'r-', label='Theta Weight')
    
    # Add labels and legend
    plt.title('Greek Sentiment Component Weight History')
    plt.xlabel('Adjustment Step')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    weights_plot_file = os.path.join(output_dir, 'weights_history_test.png')
    plt.savefig(weights_plot_file)
    plt.close()
    
    logger.info(f"Saved weights history plot to {weights_plot_file}")

def test_greek_regime_classifier(data, config, output_dir):
    """
    Test the GreekRegimeClassifier.
    
    Args:
        data (DataFrame): Sample data
        config (ConfigParser): Configuration
        output_dir (str): Output directory
    """
    logger.info("Testing GreekRegimeClassifier")
    
    # Check if Combined_Sentiment column exists
    if 'Combined_Sentiment' not in data.columns:
        logger.warning("Combined_Sentiment column not found, adding it")
        
        # Check if component sentiment columns exist
        component_columns = ['Delta_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment']
        missing_columns = [col for col in component_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning("Component sentiment columns not found, creating random Combined_Sentiment")
            data['Combined_Sentiment'] = np.random.normal(0, 1, len(data))
    
    # Process Greek sentiment
    processed_data = process_greek_sentiment(data, config)
    
    # Create classifier
    classifier = GreekRegimeClassifier(config)
    
    # Classify regimes
    classified_data = classifier.classify_regimes(processed_data)
    
    # Plot regime distribution
    regime_dist_file = os.path.join(output_dir, 'regime_distribution_test.png')
    classifier.plot_regime_distribution(classified_data, regime_dist_file)
    
    # Plot regime transitions
    regime_trans_file = os.path.join(output_dir, 'regime_transitions_test.png')
    classifier.plot_regime_transitions(classified_data, regime_trans_file)
    
    # Summarize regimes
    regime_summary = classifier.summarize_regimes(classified_data)
    
    return classified_data

def create_test_dashboard(data, output_dir):
    """
    Create a test dashboard.
    
    Args:
        data (DataFrame): Processed data
        output_dir (str): Output directory
    """
    logger.info("Creating test dashboard")
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Regime distribution
    if 'Market_Regime' in data.columns:
        plt.subplot(2, 2, 1)
        regime_counts = data['Market_Regime'].value_counts()
        sns.barplot(x=regime_counts.index, y=regime_counts.values)
        plt.title('Greek Market Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    # Plot 2: Sentiment components
    if all(col in data.columns for col in ['Delta_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment']):
        plt.subplot(2, 2, 2)
        
        # Get a sample of data points for clarity
        sample_size = min(100, len(data))
        sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
        sample_data = data.iloc[sample_indices]
        
        plt.plot(range(len(sample_data)), sample_data['Delta_Sentiment'], 'b-', label='Delta')
        plt.plot(range(len(sample_data)), sample_data['Vega_Sentiment'], 'g-', label='Vega')
        plt.plot(range(len(sample_data)), sample_data['Theta_Sentiment'], 'r-', label='Theta')
        
        plt.title('Greek Sentiment Components')
        plt.xlabel('Sample Index')
        plt.ylabel('Sentiment Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Combined sentiment
    if 'Combined_Sentiment' in data.columns:
        plt.subplot(2, 2, 3)
        
        # Get a sample of data points for clarity
        sample_size = min(100, len(data))
        sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
        sample_data = data.iloc[sample_indices]
        
        plt.plot(range(len(sample_data)), sample_data['Combined_Sentiment'], 'b-')
        
        plt.title('Combined Greek Sentiment')
        plt.xlabel('Sample Index')
        plt.ylabel('Sentiment Value')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Sentiment distribution
    if 'Combined_Sentiment' in data.columns:
        plt.subplot(2, 2, 4)
        
        sns.histplot(data['Combined_Sentiment'], kde=True)
        
        plt.title('Greek Sentiment Distribution')
        plt.xlabel('Sentiment Value')
        plt.ylabel('Frequency')
    
    # Add title to the figure
    plt.suptitle('Greek Market Regime Test Dashboard', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save dashboard
    dashboard_file = os.path.join(output_dir, 'test_dashboard.png')
    plt.savefig(dashboard_file)
    plt.close()
    
    logger.info(f"Saved test dashboard to {dashboard_file}")

def save_test_results(results, output_dir):
    """
    Save test results to file.
    
    Args:
        results (dict): Test results
        output_dir (str): Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(output_dir, 'test_results.json')
    
    # Convert non-serializable objects to strings
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved test results to {results_file}")

def main():
    """
    Main function for testing the implementation.
    """
    # Create test configuration
    config = create_test_config()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('../output/tests', f'test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test configuration
    config_file = save_test_config(config, output_dir)
    
    # Generate sample data
    sample_data = generate_sample_data(1000)
    
    # Save sample data
    sample_data_file = os.path.join(output_dir, 'sample_data.csv')
    sample_data.to_csv(sample_data_file, index=False)
    
    # Test GreekDynamicWeightAdjuster
    test_greek_dynamic_weight_adjuster(sample_data, config, output_dir)
    
    # Test GreekRegimeClassifier
    classified_data = test_greek_regime_classifier(sample_data, config, output_dir)
    
    # Save processed data
    processed_data_file = os.path.join(output_dir, 'processed_sample_data.csv')
    classified_data.to_csv(processed_data_file, index=False)
    
    # Create test dashboard
    create_test_dashboard(classified_data, output_dir)
    
    # Save test results
    test_results = {
        'timestamp': timestamp,
        'config_file': config_file,
        'sample_data_file': sample_data_file,
        'processed_data_file': processed_data_file,
        'output_dir': output_dir,
        'num_samples': len(sample_data),
        'num_regimes': config.getint('market_regime', 'num_regimes')
    }
    save_test_results(test_results, output_dir)
    
    logger.info(f"Tests completed successfully. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
