"""
Consolidated Greek Sentiment Analysis Module

This module implements Greek sentiment analysis for options trading,
focusing on tracking aggregate opening values for Vega, Delta, and Theta
and calculating minute-to-minute changes to build a dynamic sentiment gauge.

Features:
- Real-time tracking of Greeks changes from opening values
- Aggregate opening values as baseline reference points
- Minute-to-minute changes calculation
- Aggregation of changes for calls vs. puts
- Weighted sentiment score calculation
- Dynamic weight adjustment based on historical performance
- Threshold-based sentiment classification
- DTE-specific calculations
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Setup logging
logger = logging.getLogger(__name__)

class GreekSentimentAnalysis:
    """
    Consolidated Greek Sentiment Analysis.
    
    This class implements Greek sentiment analysis for options trading,
    focusing on tracking aggregate opening values for Vega, Delta, and Theta
    and calculating minute-to-minute changes to build a dynamic sentiment gauge.
    
    Implementation follows specifications in Greek_sentiment.md with aggregate
    opening values as baseline reference points and dynamic weight adjustment.
    """
    
    def __init__(self, config=None):
        """
        Initialize Greek Sentiment Analysis.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Expiry weightage
        self.expiry_weights = {
            'near': float(self.config.get('near_expiry_weight', 0.70)),
            'next': float(self.config.get('next_expiry_weight', 0.30))
        }
        
        # Default weight in market regime classification (40%)
        self.default_weight = float(self.config.get('default_weight', 0.40))
        
        # Greek component weights - these will be dynamically adjusted
        self.vega_weight = float(self.config.get('vega_weight', 0.50))
        self.delta_weight = float(self.config.get('delta_weight', 0.40))
        self.theta_weight = float(self.config.get('theta_weight', 0.10))
        
        # Sentiment thresholds - these will be dynamically adjusted
        self.sentiment_thresholds = {
            'strong_bullish': float(self.config.get('strong_bullish_threshold', 0.5)),
            'mild_bullish': float(self.config.get('mild_bullish_threshold', 0.2)),
            'neutral_upper': float(self.config.get('neutral_upper_threshold', 0.1)),
            'neutral_lower': float(self.config.get('neutral_lower_threshold', -0.1)),
            'mild_bearish': float(self.config.get('mild_bearish_threshold', -0.2)),
            'strong_bearish': float(self.config.get('strong_bearish_threshold', -0.5))
        }
        
        # Dynamic weight adjustment parameters
        self.use_dynamic_weights = bool(self.config.get('use_dynamic_weights', True))
        self.learning_rate = float(self.config.get('learning_rate', 0.1))
        self.window_size = int(self.config.get('window_size', 30))
        
        # Weight history for tracking
        self.weight_history = {
            'vega_weight': [self.vega_weight],
            'delta_weight': [self.delta_weight],
            'theta_weight': [self.theta_weight]
        }
        
        # Performance metrics history
        self.performance_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        logger.info(f"Initialized Greek Sentiment Analysis with default weight {self.default_weight}")
        logger.info(f"Using expiry weights: near={self.expiry_weights['near']}, next={self.expiry_weights['next']}")
        logger.info(f"Initial Greek weights: vega={self.vega_weight}, delta={self.delta_weight}, theta={self.theta_weight}")
        logger.info(f"Dynamic weight adjustment: {self.use_dynamic_weights}")
    
    def analyze_greek_sentiment(self, data):
        """
        Analyze Greek sentiment from options data.
        
        Args:
            data (pd.DataFrame): Options data with Greeks
            
        Returns:
            pd.DataFrame: Data with Greek sentiment analysis
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['datetime', 'option_type', 'delta', 'vega', 'theta']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to find alternative column names
            column_mapping = {
                'datetime': ['timestamp', 'date_time', 'time'],
                'option_type': ['type', 'call_put', 'cp'],
                'delta': ['Delta', 'DELTA', 'delta_value'],
                'vega': ['Vega', 'VEGA', 'vega_value'],
                'theta': ['Theta', 'THETA', 'theta_value']
            }
            
            for missing_col in missing_columns:
                for alt_col in column_mapping.get(missing_col, []):
                    if alt_col in df.columns:
                        df[missing_col] = df[alt_col]
                        logger.info(f"Using {alt_col} as {missing_col}")
                        break
            
            # Check again after mapping
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns after mapping: {missing_columns}")
                return df
        
        # Ensure datetime is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except:
                logger.warning("Failed to convert datetime column to datetime format")
                return df
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Get unique dates
        dates = df['datetime'].dt.date.unique()
        
        # Process each date separately
        result_dfs = []
        
        for date in dates:
            # Get data for this date
            date_df = df[df['datetime'].dt.date == date].copy()
            
            # Process this date
            date_result = self._process_date(date_df)
            
            # Append to results
            result_dfs.append(date_result)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=False)
        
        # Sort by datetime
        result = result.sort_values('datetime')
        
        return result
    
    def _process_date(self, date_df):
        """
        Process data for a single date.
        
        Args:
            date_df (pd.DataFrame): Data for a single date
            
        Returns:
            pd.DataFrame: Processed data with Greek sentiment
        """
        # Make a copy
        df = date_df.copy()
        
        # Get opening values for Greeks
        opening_values = self._get_opening_values(df)
        
        # Calculate minute-to-minute changes
        df = self._calculate_minute_changes(df, opening_values)
        
        # Calculate sentiment components
        df = self._calculate_sentiment_components(df)
        
        # Calculate sentiment score
        df = self._calculate_sentiment_score(df)
        
        # Classify sentiment
        df = self._classify_sentiment(df)
        
        # If dynamic weights are enabled and we have actual regime data
        if self.use_dynamic_weights and 'actual_regime' in df.columns:
            # Update weights based on performance
            df = self._update_weights(df)
        
        return df
    
    def _get_opening_values(self, df):
        """
        Get opening values for Greeks.
        
        Args:
            df (pd.DataFrame): Data for a single date
            
        Returns:
            dict: Opening values for Greeks
        """
        # Get the first timestamp
        first_timestamp = df['datetime'].min()
        
        # Get data for the first timestamp
        opening_data = df[df['datetime'] == first_timestamp]
        
        # Calculate aggregate opening values
        opening_values = {
            'call_delta': opening_data[opening_data['option_type'] == 'call']['delta'].sum(),
            'put_delta': opening_data[opening_data['option_type'] == 'put']['delta'].sum(),
            'call_vega': opening_data[opening_data['option_type'] == 'call']['vega'].sum(),
            'put_vega': opening_data[opening_data['option_type'] == 'put']['vega'].sum(),
            'call_theta': opening_data[opening_data['option_type'] == 'call']['theta'].sum(),
            'put_theta': opening_data[opening_data['option_type'] == 'put']['theta'].sum()
        }
        
        logger.info(f"Opening values: {opening_values}")
        
        return opening_values
    
    def _calculate_minute_changes(self, df, opening_values):
        """
        Calculate minute-to-minute changes from opening values.
        
        Args:
            df (pd.DataFrame): Data for a single date
            opening_values (dict): Opening values for Greeks
            
        Returns:
            pd.DataFrame: Data with minute changes
        """
        # Get unique timestamps
        timestamps = df['datetime'].unique()
        
        # Initialize result columns
        df['Delta_Change_Call'] = 0.0
        df['Delta_Change_Put'] = 0.0
        df['Vega_Change_Call'] = 0.0
        df['Vega_Change_Put'] = 0.0
        df['Theta_Change_Call'] = 0.0
        df['Theta_Change_Put'] = 0.0
        
        # Process each timestamp
        for timestamp in timestamps:
            # Get data for this timestamp
            timestamp_data = df[df['datetime'] == timestamp]
            
            # Calculate aggregate values for this timestamp
            current_values = {
                'call_delta': timestamp_data[timestamp_data['option_type'] == 'call']['delta'].sum(),
                'put_delta': timestamp_data[timestamp_data['option_type'] == 'put']['delta'].sum(),
                'call_vega': timestamp_data[timestamp_data['option_type'] == 'call']['vega'].sum(),
                'put_vega': timestamp_data[timestamp_data['option_type'] == 'put']['vega'].sum(),
                'call_theta': timestamp_data[timestamp_data['option_type'] == 'call']['theta'].sum(),
                'put_theta': timestamp_data[timestamp_data['option_type'] == 'put']['theta'].sum()
            }
            
            # Calculate changes from opening values
            delta_change_call = current_values['call_delta'] - opening_values['call_delta']
            delta_change_put = current_values['put_delta'] - opening_values['put_delta']
            vega_change_call = current_values['call_vega'] - opening_values['call_vega']
            vega_change_put = current_values['put_vega'] - opening_values['put_vega']
            theta_change_call = current_values['call_theta'] - opening_values['call_theta']
            theta_change_put = current_values['put_theta'] - opening_values['put_theta']
            
            # Update result columns for this timestamp
            df.loc[df['datetime'] == timestamp, 'Delta_Change_Call'] = delta_change_call
            df.loc[df['datetime'] == timestamp, 'Delta_Change_Put'] = delta_change_put
            df.loc[df['datetime'] == timestamp, 'Vega_Change_Call'] = vega_change_call
            df.loc[df['datetime'] == timestamp, 'Vega_Change_Put'] = vega_change_put
            df.loc[df['datetime'] == timestamp, 'Theta_Change_Call'] = theta_change_call
            df.loc[df['datetime'] == timestamp, 'Theta_Change_Put'] = theta_change_put
        
        return df
    
    def _calculate_sentiment_components(self, df):
        """
        Calculate sentiment components from Greek changes.
        
        Args:
            df (pd.DataFrame): Data with minute changes
            
        Returns:
            pd.DataFrame: Data with sentiment components
        """
        # Get unique timestamps
        timestamps = df['datetime'].unique()
        
        # Initialize result columns
        df['Delta_Component'] = 0.0
        df['Vega_Component'] = 0.0
        df['Theta_Component'] = 0.0
        
        # Process each timestamp
        for timestamp in timestamps:
            # Get data for this timestamp
            timestamp_data = df[df['datetime'] == timestamp]
            
            # Get the first row for this timestamp
            first_row = timestamp_data.iloc[0]
            
            # Calculate Delta component
            # Increasing call delta is bullish, increasing put delta is bearish
            delta_component = (first_row['Delta_Change_Call'] - first_row['Delta_Change_Put']) / 100.0
            
            # Calculate Vega component
            # Increasing call vega (call options being bought) is bullish
            # Increasing put vega (put options being bought) is bearish
            vega_component = (first_row['Vega_Change_Call'] - first_row['Vega_Change_Put']) / 100.0
            
            # Calculate Theta component
            # Increasing call theta (call options being sold) is bearish
            # Increasing put theta (put options being sold) is bullish
            theta_component = (first_row['Theta_Change_Put'] - first_row['Theta_Change_Call']) / 100.0
            
            # Update result columns for this timestamp
            df.loc[df['datetime'] == timestamp, 'Delta_Component'] = delta_component
            df.loc[df['datetime'] == timestamp, 'Vega_Component'] = vega_component
            df.loc[df['datetime'] == timestamp, 'Theta_Component'] = theta_component
        
        return df
    
    def _calculate_sentiment_score(self, df):
        """
        Calculate sentiment score from components.
        
        Args:
            df (pd.DataFrame): Data with sentiment components
            
        Returns:
            pd.DataFrame: Data with sentiment score
        """
        # Get unique timestamps
        timestamps = df['datetime'].unique()
        
        # Initialize result column
        df['Sentiment_Score'] = 0.0
        
        # Process each timestamp
        for timestamp in timestamps:
            # Get data for this timestamp
            timestamp_data = df[df['datetime'] == timestamp]
            
            # Get the first row for this timestamp
            first_row = timestamp_data.iloc[0]
            
            # Calculate weighted sentiment score
            sentiment_score = (
                self.delta_weight * first_row['Delta_Component'] +
                self.vega_weight * first_row['Vega_Component'] +
                self.theta_weight * first_row['Theta_Component']
            )
            
            # Update result column for this timestamp
            df.loc[df['datetime'] == timestamp, 'Sentiment_Score'] = sentiment_score
        
        return df
    
    def _classify_sentiment(self, df):
        """
        Classify sentiment based on score.
        
        Args:
            df (pd.DataFrame): Data with sentiment score
            
        Returns:
            pd.DataFrame: Data with classified sentiment
        """
        # Initialize result column
        df['Greek_Sentiment'] = 'Neutral'
        
        # Classify sentiment
        df.loc[df['Sentiment_Score'] > self.sentiment_thresholds['strong_bullish'], 'Greek_Sentiment'] = 'Strong_Bullish'
        df.loc[(df['Sentiment_Score'] > self.sentiment_thresholds['mild_bullish']) & 
               (df['Sentiment_Score'] <= self.sentiment_thresholds['strong_bullish']), 'Greek_Sentiment'] = 'Mild_Bullish'
        df.loc[(df['Sentiment_Score'] > self.sentiment_thresholds['neutral_upper']) & 
               (df['Sentiment_Score'] <= self.sentiment_thresholds['mild_bullish']), 'Greek_Sentiment'] = 'Sideways_To_Bullish'
        df.loc[(df['Sentiment_Score'] > self.sentiment_thresholds['neutral_lower']) & 
               (df['Sentiment_Score'] <= self.sentiment_thresholds['neutral_upper']), 'Greek_Sentiment'] = 'Neutral'
        df.loc[(df['Sentiment_Score'] <= self.sentiment_thresholds['neutral_lower']) & 
               (df['Sentiment_Score'] > self.sentiment_thresholds['mild_bearish']), 'Greek_Sentiment'] = 'Sideways_To_Bearish'
        df.loc[(df['Sentiment_Score'] <= self.sentiment_thresholds['mild_bearish']) & 
               (df['Sentiment_Score'] > self.sentiment_thresholds['strong_bearish']), 'Greek_Sentiment'] = 'Mild_Bearish'
        df.loc[df['Sentiment_Score'] <= self.sentiment_thresholds['strong_bearish'], 'Greek_Sentiment'] = 'Strong_Bearish'
        
        # Add confidence level based on distance from thresholds
        df['Sentiment_Confidence'] = 0.5  # Default medium confidence
        
        # Higher confidence for values far from thresholds
        df.loc[df['Sentiment_Score'] > self.sentiment_thresholds['strong_bullish'] * 1.5, 'Sentiment_Confidence'] = 0.9
        df.loc[df['Sentiment_Score'] < self.sentiment_thresholds['strong_bearish'] * 1.5, 'Sentiment_Confidence'] = 0.9
        
        # Lower confidence for values close to thresholds
        threshold_margin = 0.05
        for threshold in ['strong_bullish', 'mild_bullish', 'neutral_upper', 'neutral_lower', 'mild_bearish', 'strong_bearish']:
            threshold_value = self.sentiment_thresholds[threshold]
            df.loc[abs(df['Sentiment_Score'] - threshold_value) < threshold_margin, 'Sentiment_Confidence'] = 0.3
        
        return df
    
    def _update_weights(self, df):
        """
        Update weights based on performance.
        
        Args:
            df (pd.DataFrame): Data with sentiment and actual regime
            
        Returns:
            pd.DataFrame: Data with updated weights
        """
        # Check if we have enough data
        if len(df) < self.window_size:
            logger.warning(f"Not enough data for weight update: {len(df)} < {self.window_size}")
            return df
        
        # Get unique timestamps
        timestamps = df['datetime'].unique()
        
        # Initialize weight columns
        df['vega_weight'] = self.vega_weight
        df['delta_weight'] = self.delta_weight
        df['theta_weight'] = self.theta_weight
        
        # Process in windows
        for i in range(self.window_size, len(timestamps), self.window_size):
            # Get window data
            window_timestamps = timestamps[i-self.window_size:i]
            window_data = df[df['datetime'].isin(window_timestamps)].copy()
            
            # Prepare training data
            X = window_data[['Delta_Component', 'Vega_Component', 'Theta_Component']]
            y = window_data['actual_regime']
            
            # Optimize weights
            optimized_weights = self._optimize_weights(X, y)
            
            # Update weights
            self.vega_weight = optimized_weights['vega_weight']
            self.delta_weight = optimized_weights['delta_weight']
            self.theta_weight = optimized_weights['theta_weight']
            
            # Update weight history
            self.weight_history['vega_weight'].append(self.vega_weight)
            self.weight_history['delta_weight'].append(self.delta_weight)
            self.weight_history['theta_weight'].append(self.theta_weight)
            
            # Update weight columns for future timestamps
            future_timestamps = timestamps[i:]
            df.loc[df['datetime'].isin(future_timestamps), 'vega_weight'] = self.vega_weight
            df.loc[df['datetime'].isin(future_timestamps), 'delta_weight'] = self.delta_weight
            df.loc[df['datetime'].isin(future_timestamps), 'theta_weight'] = self.theta_weight
            
            # Recalculate sentiment score with new weights
            for timestamp in future_timestamps:
                # Get data for this timestamp
                timestamp_data = df[df['datetime'] == timestamp]
                
                # Get the first row for this timestamp
                first_row = timestamp_data.iloc[0]
                
                # Calculate weighted sentiment score
                sentiment_score = (
                    self.delta_weight * first_row['Delta_Component'] +
                    self.vega_weight * first_row['Vega_Component'] +
                    self.theta_weight * first_row['Theta_Component']
                )
                
                # Update result column for this timestamp
                df.loc[df['datetime'] == timestamp, 'Sentiment_Score'] = sentiment_score
            
            # Reclassify sentiment
            df = self._classify_sentiment(df)
        
        return df
    
    def _optimize_weights(self, X, y):
        """
        Optimize weights for Greek sentiment components.
        
        Args:
            X (pd.DataFrame): Features (Delta_Component, Vega_Component, Theta_Component)
            y (pd.Series): Target (actual_regime)
            
        Returns:
            dict: Optimized weights
        """
        # Initial weights
        initial_weights = [self.delta_weight, self.vega_weight, self.theta_weight]
        
        # Bounds for weights (0 to 1)
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Constraint: sum of weights = 1
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        try:
            # Optimize weights
            result = minimize(
                self._objective_function,
                initial_weights,
                args=(X, y),
                bounds=bounds,
                constraints=constraint,
                method='SLSQP'
            )
            
            # Get optimized weights
            optimized_weights = result.x
            
            # Normalize weights to sum to 1
            optimized_weights = optimized_weights / np.sum(optimized_weights)
            
            # Create weights dictionary
            weights = {
                'delta_weight': optimized_weights[0],
                'vega_weight': optimized_weights[1],
                'theta_weight': optimized_weights[2]
            }
            
            logger.info(f"Optimized weights: {weights}")
            
            return weights
        
        except Exception as e:
            logger.error(f"Error optimizing weights: {str(e)}")
            
            # Return current weights
            return {
                'delta_weight': self.delta_weight,
                'vega_weight': self.vega_weight,
                'theta_weight': self.theta_weight
            }
    
    def _objective_function(self, weights, X, y_true):
        """
        Objective function for weight optimization.
        
        Args:
            weights (list): Weights for Delta, Vega, and Theta components
            X (pd.DataFrame): Features
            y_true (pd.Series): True labels
            
        Returns:
            float: Negative accuracy (to be minimized)
        """
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted sentiment score
        sentiment_score = (
            weights[0] * X['Delta_Component'] +
            weights[1] * X['Vega_Component'] +
            weights[2] * X['Theta_Component']
        )
        
        # Classify sentiment
        y_pred = pd.Series(index=y_true.index)
        y_pred[sentiment_score > self.sentiment_thresholds['strong_bullish']] = 'Strong_Bullish'
        y_pred[(sentiment_score > self.sentiment_thresholds['mild_bullish']) & 
               (sentiment_score <= self.sentiment_thresholds['strong_bullish'])] = 'Mild_Bullish'
        y_pred[(sentiment_score > self.sentiment_thresholds['neutral_lower']) & 
               (sentiment_score <= self.sentiment_thresholds['neutral_upper'])] = 'Neutral'
        y_pred[(sentiment_score <= self.sentiment_thresholds['mild_bearish']) & 
               (sentiment_score > self.sentiment_thresholds['strong_bearish'])] = 'Mild_Bearish'
        y_pred[sentiment_score <= self.sentiment_thresholds['strong_bearish']] = 'Strong_Bearish'
        
        # Add sideways classifications
        y_pred[(sentiment_score > self.sentiment_thresholds['neutral_upper']) & 
               (sentiment_score <= self.sentiment_thresholds['mild_bullish'])] = 'Sideways_To_Bullish'
        y_pred[(sentiment_score <= self.sentiment_thresholds['neutral_lower']) & 
               (sentiment_score > self.sentiment_thresholds['mild_bearish'])] = 'Sideways_To_Bearish'
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Return negative accuracy (to be minimized)
        return -accuracy
    
    def visualize_sentiment(self, data, output_dir):
        """
        Visualize Greek sentiment analysis results.
        
        Args:
            data (pd.DataFrame): Data with Greek sentiment analysis
            output_dir (str): Output directory for visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Sentiment distribution
            plt.figure(figsize=(12, 6))
            sentiment_counts = data['Greek_Sentiment'].value_counts()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
            plt.title('Greek Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
            plt.close()
            
            # Sentiment score over time
            plt.figure(figsize=(12, 6))
            plt.plot(data['datetime'], data['Sentiment_Score'])
            plt.title('Sentiment Score Over Time')
            plt.xlabel('Time')
            plt.ylabel('Sentiment Score')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sentiment_score_time.png'))
            plt.close()
            
            # Component contributions
            plt.figure(figsize=(12, 6))
            plt.plot(data['datetime'], data['Delta_Component'], label='Delta')
            plt.plot(data['datetime'], data['Vega_Component'], label='Vega')
            plt.plot(data['datetime'], data['Theta_Component'], label='Theta')
            plt.title('Component Contributions Over Time')
            plt.xlabel('Time')
            plt.ylabel('Component Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'component_contributions.png'))
            plt.close()
            
            # If we have dynamic weights
            if 'vega_weight' in data.columns:
                # Weight evolution
                plt.figure(figsize=(12, 6))
                plt.plot(data['datetime'], data['vega_weight'], label='Vega Weight')
                plt.plot(data['datetime'], data['delta_weight'], label='Delta Weight')
                plt.plot(data['datetime'], data['theta_weight'], label='Theta Weight')
                plt.title('Weight Evolution Over Time')
                plt.xlabel('Time')
                plt.ylabel('Weight')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'weight_evolution.png'))
                plt.close()
            
            logger.info(f"Saved visualizations to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error visualizing sentiment: {str(e)}")
