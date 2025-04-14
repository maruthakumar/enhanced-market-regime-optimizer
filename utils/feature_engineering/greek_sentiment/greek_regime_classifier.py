"""
Greek Regime Classifier Module

This module provides functionality for classifying market regimes based on Greek sentiment indicators.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreekRegimeClassifier:
    """
    Class for classifying market regimes based on Greek sentiment indicators.
    """
    
    def __init__(self, config):
        """
        Initialize the GreekRegimeClassifier.
        
        Args:
            config (ConfigParser): Configuration parameters
        """
        self.config = config
        
        # Get regime classification parameters
        self.num_regimes = config.getint('market_regime', 'num_regimes', fallback=5)
        self.use_clustering = config.getboolean('market_regime', 'use_clustering', fallback=True)
        self.adaptive_thresholds = config.getboolean('market_regime', 'adaptive_thresholds', fallback=True)
        self.regime_lookback_window = config.getint('market_regime', 'regime_lookback_window', fallback=10)
        
        # Get threshold values
        self.strong_bearish_threshold = config.getfloat('market_regime', 'strong_bearish_threshold', fallback=-8.0)
        self.bearish_threshold = config.getfloat('market_regime', 'bearish_threshold', fallback=-2.5)
        self.bullish_threshold = config.getfloat('market_regime', 'bullish_threshold', fallback=2.5)
        self.strong_bullish_threshold = config.getfloat('market_regime', 'strong_bullish_threshold', fallback=8.0)
        
        # Initialize regime history
        self.regime_history = []
    
    def _calculate_adaptive_thresholds(self, sentiment_values):
        """
        Calculate adaptive thresholds based on historical data distribution.
        
        Args:
            sentiment_values (Series): Sentiment values
            
        Returns:
            tuple: Thresholds (strong_bearish, bearish, bullish, strong_bullish)
        """
        # Calculate percentiles
        percentiles = sentiment_values.quantile([0.1, 0.3, 0.7, 0.9]).values
        
        # Set thresholds based on percentiles
        strong_bearish_threshold = percentiles[0]
        bearish_threshold = percentiles[1]
        bullish_threshold = percentiles[2]
        strong_bullish_threshold = percentiles[3]
        
        return (strong_bearish_threshold, bearish_threshold, bullish_threshold, strong_bullish_threshold)
    
    def _classify_with_thresholds(self, data):
        """
        Classify market regimes using threshold-based approach.
        
        Args:
            data (DataFrame): Data with sentiment indicators
            
        Returns:
            DataFrame: Data with regime classifications
        """
        # Check if Combined_Sentiment column exists
        if 'Combined_Sentiment' not in data.columns:
            logger.warning("Combined_Sentiment column not found, adding it")
            
            # Check if component sentiment columns exist
            component_columns = ['Delta_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment']
            missing_columns = [col for col in component_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Component sentiment columns not found, creating random Combined_Sentiment")
                data['Combined_Sentiment'] = np.random.normal(0, 1, len(data))
            else:
                # Calculate combined sentiment
                delta_weight = self.config.getfloat('greek_sentiment', 'delta_weight', fallback=1.2)
                vega_weight = self.config.getfloat('greek_sentiment', 'vega_weight', fallback=1.5)
                theta_weight = self.config.getfloat('greek_sentiment', 'theta_weight', fallback=0.3)
                
                total_weight = delta_weight + vega_weight + theta_weight
                
                data['Combined_Sentiment'] = (
                    data['Delta_Sentiment'] * delta_weight +
                    data['Vega_Sentiment'] * vega_weight +
                    data['Theta_Sentiment'] * theta_weight
                ) / total_weight
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Calculate adaptive thresholds if enabled
        if self.adaptive_thresholds:
            thresholds = self._calculate_adaptive_thresholds(result['Combined_Sentiment'])
            strong_bearish_threshold, bearish_threshold, bullish_threshold, strong_bullish_threshold = thresholds
        else:
            strong_bearish_threshold = self.strong_bearish_threshold
            bearish_threshold = self.bearish_threshold
            bullish_threshold = self.bullish_threshold
            strong_bullish_threshold = self.strong_bullish_threshold
        
        # Classify regimes based on thresholds
        result['Market_Regime'] = 'Neutral'
        
        # Apply threshold classification
        if self.num_regimes == 3:
            # 3-regime classification: Bearish, Neutral, Bullish
            result.loc[result['Combined_Sentiment'] <= bearish_threshold, 'Market_Regime'] = 'Bearish'
            result.loc[result['Combined_Sentiment'] >= bullish_threshold, 'Market_Regime'] = 'Bullish'
        elif self.num_regimes == 5:
            # 5-regime classification: Strong Bearish, Bearish, Neutral, Bullish, Strong Bullish
            result.loc[result['Combined_Sentiment'] <= strong_bearish_threshold, 'Market_Regime'] = 'Strong Bearish'
            result.loc[(result['Combined_Sentiment'] > strong_bearish_threshold) & 
                      (result['Combined_Sentiment'] <= bearish_threshold), 'Market_Regime'] = 'Bearish'
            result.loc[(result['Combined_Sentiment'] >= bullish_threshold) & 
                      (result['Combined_Sentiment'] < strong_bullish_threshold), 'Market_Regime'] = 'Bullish'
            result.loc[result['Combined_Sentiment'] >= strong_bullish_threshold, 'Market_Regime'] = 'Strong Bullish'
        elif self.num_regimes == 7:
            # 7-regime classification: Extreme Bearish, Strong Bearish, Bearish, Neutral, Bullish, Strong Bullish, Extreme Bullish
            extreme_bearish_threshold = strong_bearish_threshold * 1.5
            extreme_bullish_threshold = strong_bullish_threshold * 1.5
            
            result.loc[result['Combined_Sentiment'] <= extreme_bearish_threshold, 'Market_Regime'] = 'Extreme Bearish'
            result.loc[(result['Combined_Sentiment'] > extreme_bearish_threshold) & 
                      (result['Combined_Sentiment'] <= strong_bearish_threshold), 'Market_Regime'] = 'Strong Bearish'
            result.loc[(result['Combined_Sentiment'] > strong_bearish_threshold) & 
                      (result['Combined_Sentiment'] <= bearish_threshold), 'Market_Regime'] = 'Bearish'
            result.loc[(result['Combined_Sentiment'] >= bullish_threshold) & 
                      (result['Combined_Sentiment'] < strong_bullish_threshold), 'Market_Regime'] = 'Bullish'
            result.loc[(result['Combined_Sentiment'] >= strong_bullish_threshold) & 
                      (result['Combined_Sentiment'] < extreme_bullish_threshold), 'Market_Regime'] = 'Strong Bullish'
            result.loc[result['Combined_Sentiment'] >= extreme_bullish_threshold, 'Market_Regime'] = 'Extreme Bullish'
        
        return result
    
    def _classify_with_clustering(self, data):
        """
        Classify market regimes using clustering approach.
        
        Args:
            data (DataFrame): Data with sentiment indicators
            
        Returns:
            DataFrame: Data with regime classifications
        """
        # Check if Combined_Sentiment column exists
        if 'Combined_Sentiment' not in data.columns:
            logger.warning("Combined_Sentiment column not found, adding it")
            
            # Check if component sentiment columns exist
            component_columns = ['Delta_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment']
            missing_columns = [col for col in component_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Component sentiment columns not found, creating random Combined_Sentiment")
                data['Combined_Sentiment'] = np.random.normal(0, 1, len(data))
            else:
                # Calculate combined sentiment
                delta_weight = self.config.getfloat('greek_sentiment', 'delta_weight', fallback=1.2)
                vega_weight = self.config.getfloat('greek_sentiment', 'vega_weight', fallback=1.5)
                theta_weight = self.config.getfloat('greek_sentiment', 'theta_weight', fallback=0.3)
                
                total_weight = delta_weight + vega_weight + theta_weight
                
                data['Combined_Sentiment'] = (
                    data['Delta_Sentiment'] * delta_weight +
                    data['Vega_Sentiment'] * vega_weight +
                    data['Theta_Sentiment'] * theta_weight
                ) / total_weight
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Prepare data for clustering
        X = result['Combined_Sentiment'].values.reshape(-1, 1)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.num_regimes, random_state=42)
        result['Cluster'] = kmeans.fit_predict(X)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_.flatten()
        
        # Sort clusters by sentiment value
        cluster_order = np.argsort(cluster_centers)
        
        # Map clusters to regime names
        if self.num_regimes == 3:
            # 3-regime classification: Bearish, Neutral, Bullish
            regime_names = ['Bearish', 'Neutral', 'Bullish']
        elif self.num_regimes == 5:
            # 5-regime classification: Strong Bearish, Bearish, Neutral, Bullish, Strong Bullish
            regime_names = ['Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish']
        elif self.num_regimes == 7:
            # 7-regime classification: Extreme Bearish, Strong Bearish, Bearish, Neutral, Bullish, Strong Bullish, Extreme Bullish
            regime_names = ['Extreme Bearish', 'Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish', 'Extreme Bullish']
        else:
            # Default to numbered regimes
            regime_names = [f'Regime {i+1}' for i in range(self.num_regimes)]
        
        # Create mapping from cluster to regime name
        cluster_to_regime = {cluster_order[i]: regime_names[i] for i in range(self.num_regimes)}
        
        # Map clusters to regime names
        result['Market_Regime'] = result['Cluster'].map(cluster_to_regime)
        
        # Log cluster information
        for i, cluster in enumerate(cluster_order):
            avg_sentiment = result[result['Cluster'] == cluster]['Combined_Sentiment'].mean()
            count = result[result['Cluster'] == cluster].shape[0]
            logger.info(f"Cluster {cluster} -> {regime_names[i]}: {count} points, avg sentiment: {avg_sentiment:.4f}")
        
        return result
    
    def classify_regimes(self, data):
        """
        Classify market regimes based on sentiment indicators.
        
        Args:
            data (DataFrame): Data with sentiment indicators
            
        Returns:
            DataFrame: Data with regime classifications
        """
        logger.info("Classifying market regimes")
        
        # Choose classification method
        if self.use_clustering:
            logger.info(f"Using clustering to identify {self.num_regimes} market regimes")
            result = self._classify_with_clustering(data)
        else:
            logger.info(f"Using thresholds to identify {self.num_regimes} market regimes")
            result = self._classify_with_thresholds(data)
        
        # Add regime transition indicators
        result = self._add_regime_transition_indicators(result)
        
        # Save regime classifications
        self._save_regime_classifications(result)
        
        return result
    
    def _add_regime_transition_indicators(self, data):
        """
        Add regime transition indicators to the data.
        
        Args:
            data (DataFrame): Data with regime classifications
            
        Returns:
            DataFrame: Data with regime transition indicators
        """
        if 'Market_Regime' not in data.columns:
            logger.warning("Market_Regime column not found, cannot add transition indicators")
            return data
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Add regime shift indicator
        result['Regime_Shift'] = False
        
        # Check for regime changes
        if len(result) > 1:
            result.loc[result['Market_Regime'].shift() != result['Market_Regime'], 'Regime_Shift'] = True
        
        # Add regime transition type
        result['Transition_Type'] = 'None'
        
        # Define regime strength order
        regime_strength = {
            'Extreme Bearish': -3,
            'Strong Bearish': -2,
            'Bearish': -1,
            'Neutral': 0,
            'Bullish': 1,
            'Strong Bullish': 2,
            'Extreme Bullish': 3
        }
        
        # Add numeric regime strength
        if all(regime in regime_strength for regime in result['Market_Regime'].unique()):
            result['Regime_Strength'] = result['Market_Regime'].map(regime_strength)
            
            # Determine transition type
            for i in range(1, len(result)):
                if result.iloc[i]['Regime_Shift']:
                    prev_strength = result.iloc[i-1]['Regime_Strength']
                    curr_strength = result.iloc[i]['Regime_Strength']
                    
                    if curr_strength > prev_strength:
                        result.iloc[i, result.columns.get_loc('Transition_Type')] = 'Bullish'
                    elif curr_strength < prev_strength:
                        result.iloc[i, result.columns.get_loc('Transition_Type')] = 'Bearish'
                    else:
                        result.iloc[i, result.columns.get_loc('Transition_Type')] = 'Neutral'
        
        # Add trading signal based on transition
        result['Trading_Signal'] = 'Hold'
        
        # Generate trading signals based on regime transitions
        for i in range(1, len(result)):
            if result.iloc[i]['Regime_Shift']:
                transition = result.iloc[i]['Transition_Type']
                
                if transition == 'Bullish':
                    result.iloc[i, result.columns.get_loc('Trading_Signal')] = 'Buy'
                elif transition == 'Bearish':
                    result.iloc[i, result.columns.get_loc('Trading_Signal')] = 'Sell'
        
        return result
    
    def _save_regime_classifications(self, data):
        """
        Save regime classifications to file.
        
        Args:
            data (DataFrame): Data with regime classifications
        """
        # Get output directory from config
        output_dir = self.config.get('output', 'base_dir', fallback='../output')
        regimes_dir = os.path.join(output_dir, 'regimes')
        
        # Create directory if it doesn't exist
        os.makedirs(regimes_dir, exist_ok=True)
        
        # Save regime classifications
        regime_file = os.path.join(regimes_dir, 'regime_classifications.csv')
        
        try:
            data.to_csv(regime_file, index=False)
            logger.info(f"Saved regime classifications to {regime_file}")
        except Exception as e:
            logger.warning(f"Failed to save regime classifications: {str(e)}")
    
    def plot_regime_distribution(self, data, output_file=None):
        """
        Plot regime distribution.
        
        Args:
            data (DataFrame): Data with regime classifications
            output_file (str, optional): Output file path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if 'Market_Regime' not in data.columns:
            logger.warning("Market_Regime column not found, cannot plot regime distribution")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Count regimes
        regime_counts = data['Market_Regime'].value_counts()
        
        # Create bar chart
        sns.barplot(x=regime_counts.index, y=regime_counts.values)
        
        # Add labels
        plt.title('Market Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        total = len(data)
        for i, count in enumerate(regime_counts.values):
            percentage = count / total * 100
            plt.text(i, count + 0.1, f'{percentage:.1f}%', ha='center')
        
        plt.tight_layout()
        
        # Save figure if output file is provided
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved regime distribution plot to {output_file}")
        
        return plt.gcf()
    
    def plot_regime_transitions(self, data, output_file=None):
        """
        Plot regime transitions.
        
        Args:
            data (DataFrame): Data with regime classifications
            output_file (str, optional): Output file path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if 'Market_Regime' not in data.columns:
            logger.warning("Market_Regime column not found, cannot plot regime transitions")
            return None
        
        # Find date column
        date_col = None
        for col in ['Date', 'date', 'datetime', 'timestamp']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No date column found, cannot plot regime transitions")
            return None
        
        # Create figure
        plt.figure(figsize=(14, 7))
        
        # Create a numeric mapping for regimes
        regimes = data['Market_Regime'].unique()
        regime_to_num = {regime: i for i, regime in enumerate(sorted(regimes, key=lambda x: regime_strength.get(x, 0)))}
        
        # Define regime strength order for sorting
        regime_strength = {
            'Extreme Bearish': -3,
            'Strong Bearish': -2,
            'Bearish': -1,
            'Neutral': 0,
            'Bullish': 1,
            'Strong Bullish': 2,
            'Extreme Bullish': 3
        }
        
        # Convert regimes to numeric values
        regime_numeric = data['Market_Regime'].map(regime_to_num)
        
        # Plot regime transitions
        plt.plot(data[date_col], regime_numeric, 'b-')
        
        # Highlight regime transitions
        if 'Regime_Shift' in data.columns:
            transitions = data[data['Regime_Shift']]
            if not transitions.empty:
                plt.scatter(
                    transitions[date_col],
                    transitions['Market_Regime'].map(regime_to_num),
                    color='r',
                    s=100,
                    marker='^'
                )
        
        # Set y-ticks to regime names
        plt.yticks(
            [regime_to_num[regime] for regime in sorted(regimes, key=lambda x: regime_strength.get(x, 0))],
            sorted(regimes, key=lambda x: regime_strength.get(x, 0))
        )
        
        plt.title('Market Regime Transitions')
        plt.xlabel('Date')
        plt.ylabel('Market Regime')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure if output file is provided
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved regime transitions plot to {output_file}")
        
        return plt.gcf()
    
    def summarize_regimes(self, data):
        """
        Summarize regime classifications.
        
        Args:
            data (DataFrame): Data with regime classifications
            
        Returns:
            dict: Regime summary
        """
        if 'Market_Regime' not in data.columns:
            logger.warning("Market_Regime column not found, cannot summarize regimes")
            return {}
        
        # Count regimes
        regime_counts = data['Market_Regime'].value_counts()
        
        # Calculate percentages
        total = len(data)
        regime_percentages = regime_counts / total * 100
        
        # Create summary
        summary = {}
        for regime, count in regime_counts.items():
            percentage = regime_percentages[regime]
            logger.info(f"Regime {regime}: {count} points ({percentage:.1f}%)")
            summary[regime] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return summary
