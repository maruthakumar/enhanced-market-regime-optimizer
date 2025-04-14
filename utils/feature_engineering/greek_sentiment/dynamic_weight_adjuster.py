"""
Dynamic weight adjuster module for the enhanced market regime optimizer.
This module provides functionality to dynamically adjust weights for Greek sentiment
components based on historical performance.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DynamicWeightAdjuster:
    """Class to dynamically adjust weights for Greek sentiment components based on historical performance."""
    
    def __init__(self, config):
        """
        Initialize the DynamicWeightAdjuster with configuration.
        
        Args:
            config (ConfigParser or dict): Configuration parameters
        """
        self.config = config
        
        # Handle different config types
        if isinstance(config, dict):
            self.data_dir = config.get('market_data_dir') if 'market_data_dir' in config else '../data/market_data'
            base_dir = config.get('base_dir') if 'base_dir' in config else '../output'
            self.history_window = config.get('weight_history_window', 20)
            self.learning_rate = config.get('weight_learning_rate', 0.05)
            self.min_weight = config.get('min_weight', 0.1)
            self.max_weight = config.get('max_weight', 3.0)
            self.initial_weights = {
                'delta': config.get('delta_weight', 1.0),
                'vega': config.get('vega_weight', 1.0),
                'theta': config.get('theta_weight', 0.5)
            }
        else:
            self.data_dir = config.get('market_regime', 'market_data_dir', fallback='../data/market_data')
            base_dir = config.get('output', 'base_dir', fallback='../output')
            self.history_window = config.getint('greek_sentiment', 'weight_history_window', fallback=20)
            self.learning_rate = config.getfloat('greek_sentiment', 'weight_learning_rate', fallback=0.05)
            self.min_weight = config.getfloat('greek_sentiment', 'min_weight', fallback=0.1)
            self.max_weight = config.getfloat('greek_sentiment', 'max_weight', fallback=3.0)
            self.initial_weights = {
                'delta': config.getfloat('greek_sentiment', 'delta_weight', fallback=1.0),
                'vega': config.getfloat('greek_sentiment', 'vega_weight', fallback=1.0),
                'theta': config.getfloat('greek_sentiment', 'theta_weight', fallback=0.5)
            }
            
        self.output_dir = os.path.join(base_dir, 'weights')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize weights history storage
        self.weights_history_file = os.path.join(self.output_dir, 'weights_history.json')
        self.performance_history_file = os.path.join(self.output_dir, 'performance_history.json')
        
        # Load existing weights history if available
        self.weights_history = self._load_weights_history()
        self.performance_history = self._load_performance_history()
        
        # Current weights (start with initial or latest from history)
        if self.weights_history:
            self.current_weights = self.weights_history[-1]
        else:
            self.current_weights = self.initial_weights.copy()
    
    def _load_weights_history(self):
        """
        Load weights history from file if available.
        
        Returns:
            list: History of weight configurations
        """
        if os.path.exists(self.weights_history_file):
            try:
                with open(self.weights_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading weights history: {str(e)}")
                return []
        return []
    
    def _load_performance_history(self):
        """
        Load performance history from file if available.
        
        Returns:
            list: History of performance metrics
        """
        if os.path.exists(self.performance_history_file):
            try:
                with open(self.performance_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {str(e)}")
                return []
        return []
    
    def _save_weights_history(self):
        """Save weights history to file."""
        try:
            with open(self.weights_history_file, 'w') as f:
                json.dump(self.weights_history, f, indent=2)
            logger.info(f"Saved weights history to {self.weights_history_file}")
        except Exception as e:
            logger.error(f"Error saving weights history: {str(e)}")
    
    def _save_performance_history(self):
        """Save performance history to file."""
        try:
            with open(self.performance_history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            logger.info(f"Saved performance history to {self.performance_history_file}")
        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")
    
    def get_current_weights(self):
        """
        Get current weights for Greek sentiment components.
        
        Returns:
            dict: Current weights for delta, vega, and theta
        """
        return self.current_weights
    
    def evaluate_component_performance(self, data, market_data=None, price_column='Close', return_column='Return'):
        """
        Evaluate the predictive performance of each Greek sentiment component.
        
        Args:
            data (DataFrame): Data with Greek sentiment components
            market_data (DataFrame, optional): Market price data for validation
            price_column (str): Column name for price in market_data
            return_column (str): Column name for returns in market_data
            
        Returns:
            dict: Performance metrics for each component
        """
        logger.info("Evaluating Greek sentiment component performance")
        
        # Initialize performance metrics
        performance = {
            'delta': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
        
        # Check if we have the necessary data
        required_columns = ['Delta_Sentiment', 'Vega_Sentiment']
        if not all(col in data.columns for col in required_columns):
            logger.warning("Missing required sentiment columns for performance evaluation")
            return performance
        
        # If market data is provided, we can calculate directional accuracy
        if market_data is not None and price_column in market_data.columns:
            # Ensure we have a datetime column for joining
            date_col = None
            for col in ['datetime', 'Date', 'date']:
                if col in data.columns and col in market_data.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.warning("No common date column found for joining with market data")
            else:
                # Calculate returns if not already present
                if return_column not in market_data.columns:
                    market_data[return_column] = market_data[price_column].pct_change()
                
                # Join sentiment data with market data
                merged_data = pd.merge(
                    data,
                    market_data[[date_col, return_column]],
                    on=date_col,
                    how='inner'
                )
                
                if len(merged_data) > 0:
                    # Calculate directional accuracy for Delta
                    delta_correct = ((merged_data['Delta_Sentiment'] > 0) & (merged_data[return_column] > 0)) | \
                                   ((merged_data['Delta_Sentiment'] < 0) & (merged_data[return_column] < 0))
                    performance['delta'] = delta_correct.mean()
                    
                    # For Vega, we would ideally compare with future realized volatility
                    # As a proxy, we can use absolute returns
                    abs_returns = merged_data[return_column].abs()
                    vega_correct = ((merged_data['Vega_Sentiment'] > 0) & (abs_returns > abs_returns.mean())) | \
                                  ((merged_data['Vega_Sentiment'] < 0) & (abs_returns < abs_returns.mean()))
                    performance['vega'] = vega_correct.mean()
                    
                    # For Theta, if available
                    if 'Theta_Sentiment' in merged_data.columns:
                        # Theta is related to time decay, which is harder to validate directly
                        # As a proxy, we can check if Theta correctly predicts low volatility periods
                        theta_correct = ((merged_data['Theta_Sentiment'] > 0) & (abs_returns < abs_returns.mean())) | \
                                       ((merged_data['Theta_Sentiment'] < 0) & (abs_returns > abs_returns.mean()))
                        performance['theta'] = theta_correct.mean()
        
        # If we don't have market data, use internal consistency as a metric
        else:
            # Check correlation between sentiment and subsequent sentiment change
            # This assumes that good predictors should show some autocorrelation
            for component in ['Delta_Sentiment', 'Vega_Sentiment']:
                if component in data.columns:
                    # Calculate lagged correlation
                    component_name = component.split('_')[0].lower()
                    autocorr = data[component].autocorr(lag=1)
                    if not np.isnan(autocorr):
                        # Transform correlation to 0-1 scale (0.5 is random, 1.0 is perfect)
                        performance[component_name] = (abs(autocorr) + 0.5) / 1.5
            
            # For Theta if available
            if 'Theta_Sentiment' in data.columns:
                autocorr = data['Theta_Sentiment'].autocorr(lag=1)
                if not np.isnan(autocorr):
                    performance['theta'] = (abs(autocorr) + 0.5) / 1.5
        
        logger.info(f"Component performance: Delta={performance['delta']:.4f}, Vega={performance['vega']:.4f}, Theta={performance['theta']:.4f}")
        return performance
    
    def adjust_weights(self, performance):
        """
        Adjust weights based on component performance.
        
        Args:
            performance (dict): Performance metrics for each component
            
        Returns:
            dict: Updated weights
        """
        logger.info("Adjusting weights based on component performance")
        
        # Calculate weight adjustments based on relative performance
        total_performance = sum(performance.values())
        
        if total_performance > 0:
            # Calculate target weights based on relative performance
            target_weights = {
                component: (perf / total_performance) * sum(self.current_weights.values())
                for component, perf in performance.items()
            }
            
            # Apply gradual adjustment using learning rate
            new_weights = {
                component: self.current_weights[component] + 
                          self.learning_rate * (target_weights[component] - self.current_weights[component])
                for component in self.current_weights
            }
            
            # Ensure weights stay within bounds
            new_weights = {
                component: max(min(weight, self.max_weight), self.min_weight)
                for component, weight in new_weights.items()
            }
        else:
            # If performance metrics are all zero, revert to initial weights
            new_weights = self.initial_weights.copy()
        
        # Normalize weights to ensure they sum to the same total as before
        total_old = sum(self.current_weights.values())
        total_new = sum(new_weights.values())
        
        if total_new > 0:
            new_weights = {
                component: (weight / total_new) * total_old
                for component, weight in new_weights.items()
            }
        
        # Update current weights
        self.current_weights = new_weights
        
        # Add to history (keeping only the most recent window)
        self.weights_history.append(new_weights)
        if len(self.weights_history) > self.history_window:
            self.weights_history = self.weights_history[-self.history_window:]
        
        # Add performance to history
        self.performance_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance': performance,
            'weights': new_weights
        })
        if len(self.performance_history) > self.history_window:
            self.performance_history = self.performance_history[-self.history_window:]
        
        # Save updated histories
        self._save_weights_history()
        self._save_performance_history()
        
        logger.info(f"Adjusted weights: Delta={new_weights['delta']:.4f}, Vega={new_weights['vega']:.4f}, Theta={new_weights['theta']:.4f}")
        return new_weights
    
    def update_weights_based_on_history(self, data, market_data=None):
        """
        Update weights based on historical performance.
        
        Args:
            data (DataFrame): Data with Greek sentiment components
            market_data (DataFrame, optional): Market price data for validation
            
        Returns:
            dict: Updated weights
        """
        # Evaluate component performance
        performance = self.evaluate_component_performance(data, market_data)
        
        # Adjust weights based on performance
        new_weights = self.adjust_weights(performance)
        
        return new_weights
    
    def get_weights_history_df(self):
        """
        Get weights history as a DataFrame.
        
        Returns:
            DataFrame: History of weight adjustments
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        # Extract data from performance history
        history_data = []
        for entry in self.performance_history:
            record = {
                'timestamp': entry['timestamp'],
                'delta_weight': entry['weights']['delta'],
                'vega_weight': entry['weights']['vega'],
                'theta_weight': entry['weights']['theta'],
                'delta_performance': entry['performance']['delta'],
                'vega_performance': entry['performance']['vega'],
                'theta_performance': entry['performance']['theta']
            }
            history_data.append(record)
        
        return pd.DataFrame(history_data)
    
    def plot_weights_history(self, output_file=None):
        """
        Plot the history of weight adjustments.
        
        Args:
            output_file (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot file
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get history as DataFrame
            history_df = self.get_weights_history_df()
            
            if len(history_df) < 2:
                logger.warning("Not enough history data for plotting")
                return None
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot weights
            plt.subplot(2, 1, 1)
            plt.plot(history_df['timestamp'], history_df['delta_weight'], 'b-', label='Delta Weight')
            plt.plot(history_df['timestamp'], history_df['vega_weight'], 'g-', label='Vega Weight')
            plt.plot(history_df['timestamp'], history_df['theta_weight'], 'r-', label='Theta Weight')
            plt.title('Greek Sentiment Component Weights Over Time')
            plt.ylabel('Weight Value')
            plt.legend()
            plt.grid(True)
            
            # Plot performance
            plt.subplot(2, 1, 2)
            plt.plot(history_df['timestamp'], history_df['delta_performance'], 'b--', label='Delta Performance')
            plt.plot(history_df['timestamp'], history_df['vega_performance'], 'g--', label='Vega Performance')
            plt.plot(history_df['timestamp'], history_df['theta_performance'], 'r--', label='Theta Performance')
            plt.title('Greek Sentiment Component Performance Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Performance Metric')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot if output file is specified
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = os.path.join(self.output_dir, f'weights_history_{timestamp}.png')
            
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Saved weights history plot to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error plotting weights history: {str(e)}")
            return None


def get_dynamic_weights(data, config, market_data=None):
    """
    Get dynamically adjusted weights for Greek sentiment components.
    
    Args:
        data (DataFrame): Data with Greek sentiment components
        config (ConfigParser or dict): Configuration parameters
        market_data (DataFrame, optional): Market price data for validation
        
    Returns:
        dict: Dynamically adjusted weights
    """
    adjuster = DynamicWeightAdjuster(config)
    return adjuster.update_weights_based_on_history(data, market_data)
