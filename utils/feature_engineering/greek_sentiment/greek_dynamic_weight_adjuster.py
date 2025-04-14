"""
Greek Dynamic Weight Adjuster Module

This module provides functionality for dynamically adjusting weights of Greek sentiment components
(Delta, Vega, Theta) based on historical performance.
"""

import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreekDynamicWeightAdjuster:
    """
    Class for dynamically adjusting weights of Greek sentiment components based on historical performance.
    """
    
    def __init__(self, config):
        """
        Initialize the GreekDynamicWeightAdjuster.
        
        Args:
            config (ConfigParser): Configuration parameters
        """
        self.config = config
        
        # Get initial weights from config
        self.delta_weight = config.getfloat('greek_sentiment', 'delta_weight', fallback=1.2)
        self.vega_weight = config.getfloat('greek_sentiment', 'vega_weight', fallback=1.5)
        self.theta_weight = config.getfloat('greek_sentiment', 'theta_weight', fallback=0.3)
        
        # Get weight adjustment parameters
        self.enable_dynamic_weights = config.getboolean('greek_sentiment', 'enable_dynamic_weights', fallback=True)
        self.weight_history_window = config.getint('greek_sentiment', 'weight_history_window', fallback=10)
        self.weight_learning_rate = config.getfloat('greek_sentiment', 'weight_learning_rate', fallback=0.05)
        self.min_weight = config.getfloat('greek_sentiment', 'min_weight', fallback=0.1)
        self.max_weight = config.getfloat('greek_sentiment', 'max_weight', fallback=3.0)
        
        # Initialize weight history
        self.weight_history = {
            'timestamp': [],
            'delta_weight': [],
            'vega_weight': [],
            'theta_weight': []
        }
        
        # Initialize performance history
        self.performance_history = {
            'timestamp': [],
            'delta_performance': [],
            'vega_performance': [],
            'theta_performance': []
        }
        
        # Load weight history if available
        self._load_weight_history()
    
    def _load_weight_history(self):
        """
        Load weight history from file if available.
        """
        # Get output directory from config
        output_dir = self.config.get('output', 'base_dir', fallback='../output')
        weights_dir = os.path.join(output_dir, 'weights')
        
        # Create directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)
        
        # Weight history file
        weight_history_file = os.path.join(weights_dir, 'weights_history.json')
        
        # Load weight history if file exists
        if os.path.exists(weight_history_file):
            try:
                with open(weight_history_file, 'r') as f:
                    self.weight_history = json.load(f)
                logger.info(f"Loaded weight history from {weight_history_file}")
            except Exception as e:
                logger.warning(f"Failed to load weight history: {str(e)}")
        
        # Performance history file
        performance_history_file = os.path.join(weights_dir, 'performance_history.json')
        
        # Load performance history if file exists
        if os.path.exists(performance_history_file):
            try:
                with open(performance_history_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded performance history from {performance_history_file}")
            except Exception as e:
                logger.warning(f"Failed to load performance history: {str(e)}")
    
    def _save_weight_history(self):
        """
        Save weight history to file.
        """
        # Get output directory from config
        output_dir = self.config.get('output', 'base_dir', fallback='../output')
        weights_dir = os.path.join(output_dir, 'weights')
        
        # Create directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)
        
        # Weight history file
        weight_history_file = os.path.join(weights_dir, 'weights_history.json')
        
        # Save weight history
        try:
            with open(weight_history_file, 'w') as f:
                json.dump(self.weight_history, f)
            logger.info(f"Saved weights history to {weight_history_file}")
        except Exception as e:
            logger.warning(f"Failed to save weight history: {str(e)}")
        
        # Performance history file
        performance_history_file = os.path.join(weights_dir, 'performance_history.json')
        
        # Save performance history
        try:
            with open(performance_history_file, 'w') as f:
                json.dump(self.performance_history, f)
            logger.info(f"Saved performance history to {performance_history_file}")
        except Exception as e:
            logger.warning(f"Failed to save performance history: {str(e)}")
    
    def evaluate_component_performance(self, data):
        """
        Evaluate the performance of each Greek sentiment component.
        
        Args:
            data (DataFrame): Data with Greek sentiment components and market returns
            
        Returns:
            dict: Performance scores for each component
        """
        # Check if required columns exist
        required_columns = ['Delta_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment', 'Market_Return']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required sentiment columns for performance evaluation: {missing_columns}")
            
            # If Market_Return is missing, we can't evaluate performance
            if 'Market_Return' in missing_columns:
                return {
                    'delta_performance': 0.5,
                    'vega_performance': 0.5,
                    'theta_performance': 0.5
                }
            
            # If some sentiment columns are missing, use default values
            if 'Delta_Sentiment' in missing_columns:
                data['Delta_Sentiment'] = 0.0
            if 'Vega_Sentiment' in missing_columns:
                data['Vega_Sentiment'] = 0.0
            if 'Theta_Sentiment' in missing_columns:
                data['Theta_Sentiment'] = 0.0
        
        # Calculate correlation between sentiment and returns
        delta_corr = data['Delta_Sentiment'].corr(data['Market_Return'])
        vega_corr = data['Vega_Sentiment'].corr(data['Market_Return'])
        theta_corr = data['Theta_Sentiment'].corr(data['Market_Return'])
        
        # Handle NaN values
        delta_corr = 0.0 if pd.isna(delta_corr) else delta_corr
        vega_corr = 0.0 if pd.isna(vega_corr) else vega_corr
        theta_corr = 0.0 if pd.isna(theta_corr) else theta_corr
        
        # Calculate performance scores (normalize to 0-1 range)
        delta_performance = (delta_corr + 1) / 2
        vega_performance = (vega_corr + 1) / 2
        theta_performance = (theta_corr + 1) / 2
        
        # Update performance history
        self.performance_history['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.performance_history['delta_performance'].append(delta_performance)
        self.performance_history['vega_performance'].append(vega_performance)
        self.performance_history['theta_performance'].append(theta_performance)
        
        # Limit history length
        if len(self.performance_history['timestamp']) > self.weight_history_window:
            self.performance_history['timestamp'] = self.performance_history['timestamp'][-self.weight_history_window:]
            self.performance_history['delta_performance'] = self.performance_history['delta_performance'][-self.weight_history_window:]
            self.performance_history['vega_performance'] = self.performance_history['vega_performance'][-self.weight_history_window:]
            self.performance_history['theta_performance'] = self.performance_history['theta_performance'][-self.weight_history_window:]
        
        return {
            'delta_performance': delta_performance,
            'vega_performance': vega_performance,
            'theta_performance': theta_performance
        }
    
    def adjust_weights(self, data=None):
        """
        Adjust weights based on component performance.
        
        Args:
            data (DataFrame, optional): Data with Greek sentiment components and market returns
            
        Returns:
            dict: Adjusted weights
        """
        # If dynamic weights are disabled, return current weights
        if not self.enable_dynamic_weights:
            logger.info(f"Dynamic weight adjustment is disabled")
            return {
                'delta_weight': self.delta_weight,
                'vega_weight': self.vega_weight,
                'theta_weight': self.theta_weight
            }
        
        # Evaluate component performance if data is provided
        if data is not None:
            performance = self.evaluate_component_performance(data)
        else:
            # Use average of historical performance if available
            if len(self.performance_history['timestamp']) > 0:
                performance = {
                    'delta_performance': np.mean(self.performance_history['delta_performance']),
                    'vega_performance': np.mean(self.performance_history['vega_performance']),
                    'theta_performance': np.mean(self.performance_history['theta_performance'])
                }
            else:
                # Use default performance if no history
                performance = {
                    'delta_performance': 0.5,
                    'vega_performance': 0.5,
                    'theta_performance': 0.5
                }
        
        # Calculate weight adjustments
        delta_adjustment = (performance['delta_performance'] - 0.5) * 2 * self.weight_learning_rate
        vega_adjustment = (performance['vega_performance'] - 0.5) * 2 * self.weight_learning_rate
        theta_adjustment = (performance['theta_performance'] - 0.5) * 2 * self.weight_learning_rate
        
        # Apply adjustments
        self.delta_weight += delta_adjustment
        self.vega_weight += vega_adjustment
        self.theta_weight += theta_adjustment
        
        # Ensure weights are within bounds
        self.delta_weight = max(self.min_weight, min(self.max_weight, self.delta_weight))
        self.vega_weight = max(self.min_weight, min(self.max_weight, self.vega_weight))
        self.theta_weight = max(self.min_weight, min(self.max_weight, self.theta_weight))
        
        # Update weight history
        self.weight_history['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.weight_history['delta_weight'].append(self.delta_weight)
        self.weight_history['vega_weight'].append(self.vega_weight)
        self.weight_history['theta_weight'].append(self.theta_weight)
        
        # Limit history length
        if len(self.weight_history['timestamp']) > self.weight_history_window:
            self.weight_history['timestamp'] = self.weight_history['timestamp'][-self.weight_history_window:]
            self.weight_history['delta_weight'] = self.weight_history['delta_weight'][-self.weight_history_window:]
            self.weight_history['vega_weight'] = self.weight_history['vega_weight'][-self.weight_history_window:]
            self.weight_history['theta_weight'] = self.weight_history['theta_weight'][-self.weight_history_window:]
        
        # Save weight history
        self._save_weight_history()
        
        logger.info(f"Adjusted weights: Delta={self.delta_weight:.4f}, Vega={self.vega_weight:.4f}, Theta={self.theta_weight:.4f}")
        
        return {
            'delta_weight': self.delta_weight,
            'vega_weight': self.vega_weight,
            'theta_weight': self.theta_weight
        }
    
    def get_current_weights(self):
        """
        Get current weights.
        
        Returns:
            dict: Current weights
        """
        return {
            'delta_weight': self.delta_weight,
            'vega_weight': self.vega_weight,
            'theta_weight': self.theta_weight
        }
    
    def plot_weight_history(self, output_file=None):
        """
        Plot weight history.
        
        Args:
            output_file (str, optional): Output file path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Check if weight history is available
        if len(self.weight_history['timestamp']) == 0:
            logger.warning("No weight history available for plotting")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot weight history
        plt.plot(self.weight_history['delta_weight'], 'b-', label='Delta Weight')
        plt.plot(self.weight_history['vega_weight'], 'g-', label='Vega Weight')
        plt.plot(self.weight_history['theta_weight'], 'r-', label='Theta Weight')
        
        # Add labels and legend
        plt.title('Greek Sentiment Component Weight History')
        plt.xlabel('Adjustment Step')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure if output file is provided
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved weights history plot to {output_file}")
        
        return plt.gcf()
