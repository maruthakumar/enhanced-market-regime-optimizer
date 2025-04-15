import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

# Setup logging
logger = logging.getLogger(__name__)

class DynamicWeightAdjustment:
    """
    Dynamic Weight Adjustment System.
    
    This class implements dynamic weight adjustment for market regime components,
    optimizing weights based on historical performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize Dynamic Weight Adjustment.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Window size for rolling optimization
        self.window_size = int(self.config.get('window_size', 30))
        
        # Learning rate for performance-based adjustment
        self.learning_rate = float(self.config.get('learning_rate', 0.05))
        
        # Minimum weight threshold
        self.min_weight = float(self.config.get('min_weight', 0.05))
        
        # Default component weights
        self.default_weights = {
            'greek_sentiment': 0.4,
            'trending_oi_pa': 0.3,
            'iv_skew': 0.1,
            'ema': 0.1,
            'vwap': 0.05,
            'atr': 0.05
        }
        
        # Current weights (start with defaults)
        self.current_weights = self.default_weights.copy()
        
        # Performance history
        self.performance_history = {component: [] for component in self.default_weights}
        
        logger.info(f"Initialized Dynamic Weight Adjustment with window size: {self.window_size}")
    
    def optimize_weights(self, historical_data, actual_regimes, component_signals):
        """
        Optimize weights based on historical data.
        
        Args:
            historical_data (pd.DataFrame): Historical data
            actual_regimes (pd.Series): Actual market regimes
            component_signals (dict): Dictionary of component signals
            
        Returns:
            dict: Optimized weights
        """
        # Ensure we have enough data
        if len(historical_data) < self.window_size:
            logger.warning(f"Not enough historical data for optimization (need {self.window_size}, got {len(historical_data)})")
            return self.current_weights.copy()
        
        # Get the most recent window of data
        recent_data = historical_data.iloc[-self.window_size:]
        recent_regimes = actual_regimes.iloc[-self.window_size:]
        
        # Extract component signals for the window
        window_signals = {}
        for component, signals in component_signals.items():
            if len(signals) >= self.window_size:
                window_signals[component] = signals[-self.window_size:]
            else:
                logger.warning(f"Not enough signal data for component {component}")
                window_signals[component] = [0] * self.window_size
        
        # Define objective function for optimization
        def objective(weights_array):
            # Convert weights array to dictionary
            weights = {component: weight for component, weight in zip(self.default_weights.keys(), weights_array)}
            
            # Calculate predicted regimes using these weights
            predicted_regimes = self._calculate_regimes(window_signals, weights)
            
            # Calculate F1 score (balance between precision and recall)
            f1_score = self._calculate_f1_score(recent_regimes, predicted_regimes)
            
            # Return negative F1 score (since we want to maximize it)
            return -f1_score
        
        # Initial weights
        initial_weights = [self.current_weights[component] for component in self.default_weights.keys()]
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
        
        # Bounds: each weight between min_weight and 1
        bounds = [(self.min_weight, 1) for _ in range(len(initial_weights))]
        
        # Perform optimization
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Check if optimization was successful
        if result.success:
            # Convert optimized weights array to dictionary
            optimized_weights = {
                component: max(self.min_weight, weight) 
                for component, weight in zip(self.default_weights.keys(), result.x)
            }
            
            # Normalize weights to ensure they sum to 1
            total_weight = sum(optimized_weights.values())
            optimized_weights = {
                component: weight / total_weight 
                for component, weight in optimized_weights.items()
            }
            
            logger.info(f"Optimized weights: {optimized_weights}")
            return optimized_weights
        else:
            logger.warning(f"Weight optimization failed: {result.message}")
            return self.current_weights.copy()
    
    def _calculate_regimes(self, component_signals, weights):
        """
        Calculate regimes based on component signals and weights.
        
        Args:
            component_signals (dict): Dictionary of component signals
            weights (dict): Component weights
            
        Returns:
            list: Predicted regimes
        """
        # Initialize predicted regimes
        predicted_regimes = []
        
        # Calculate weighted sum of signals for each time point
        for i in range(self.window_size):
            weighted_sum = 0
            confidence_sum = 0
            
            # Process each component
            for component in weights.keys():
                if component in component_signals:
                    # Check if we have both signal and confidence
                    if isinstance(component_signals[component][i], dict) and 'value' in component_signals[component][i] and 'confidence' in component_signals[component][i]:
                        # Extract signal value and confidence
                        signal_value = component_signals[component][i]['value']
                        signal_confidence = component_signals[component][i]['confidence']
                        
                        # Apply confidence-weighted contribution
                        weighted_sum += weights[component] * signal_value * signal_confidence
                        confidence_sum += weights[component] * signal_confidence
                    else:
                        # Use signal directly if no confidence available
                        weighted_sum += weights[component] * component_signals[component][i]
                        confidence_sum += weights[component]
            
            # Normalize by confidence if available
            if confidence_sum > 0:
                weighted_sum = weighted_sum / confidence_sum
            
            # Convert weighted sum to regime
            # This is a simplified version; actual implementation would map to specific regimes
            predicted_regimes.append(weighted_sum)
        
        return predicted_regimes
    
    def _calculate_f1_score(self, actual, predicted):
        """
        Calculate F1 score between actual and predicted regimes.
        
        Args:
            actual (pd.Series): Actual regimes
            predicted (list): Predicted regimes
            
        Returns:
            float: F1 score
        """
        # Convert to numpy arrays
        actual_array = np.array(actual)
        predicted_array = np.array(predicted)
        
        # Calculate mean squared error (simplified metric)
        mse = np.mean((actual_array - predicted_array) ** 2)
        
        # Convert to F1-like score (1 is perfect, 0 is worst)
        f1_like = 1 / (1 + mse)
        
        return f1_like
    
    def adjust_thresholds(self, historical_data, volatility_metric='ATR'):
        """
        Dynamically adjust thresholds based on market volatility.
        
        Args:
            historical_data (pd.DataFrame): Historical data
            volatility_metric (str): Column name for volatility metric
            
        Returns:
            dict: Adjusted thresholds
        """
        # Ensure volatility metric exists
        if volatility_metric not in historical_data.columns:
            logger.warning(f"Volatility metric {volatility_metric} not found in data")
            return {
                'directional': {'strong': 0.5, 'mild': 0.2, 'neutral': 0.1},
                'volatility': {'high': 0.7, 'low': 0.3}
            }
        
        # Get recent volatility data
        recent_volatility = historical_data[volatility_metric].iloc[-self.window_size:]
        
        # Calculate percentiles
        volatility_percentile = np.percentile(recent_volatility, [30, 50, 70, 90])
        
        # Adjust thresholds based on current market volatility
        current_volatility = recent_volatility.iloc[-1]
        
        # Base thresholds
        base_thresholds = {
            'directional': {'strong': 0.5, 'mild': 0.2, 'neutral': 0.1},
            'volatility': {'high': 0.7, 'low': 0.3}
        }
        
        # Adjust based on volatility percentile
        if current_volatility > volatility_percentile[3]:  # Above 90th percentile
            # In very high volatility, widen thresholds
            adjusted_thresholds = {
                'directional': {
                    'strong': base_thresholds['directional']['strong'] * 1.5,
                    'mild': base_thresholds['directional']['mild'] * 1.3,
                    'neutral': base_thresholds['directional']['neutral'] * 1.2
                },
                'volatility': {
                    'high': base_thresholds['volatility']['high'] * 0.9,  # Lower threshold to catch more high volatility
                    'low': base_thresholds['volatility']['low'] * 0.8
                }
            }
        elif current_volatility > volatility_percentile[2]:  # Above 70th percentile
            # In high volatility, slightly widen thresholds
            adjusted_thresholds = {
                'directional': {
                    'strong': base_thresholds['directional']['strong'] * 1.3,
                    'mild': base_thresholds['directional']['mild'] * 1.2,
                    'neutral': base_thresholds['directional']['neutral'] * 1.1
                },
                'volatility': {
                    'high': base_thresholds['volatility']['high'] * 0.95,
                    'low': base_thresholds['volatility']['low'] * 0.9
                }
            }
        elif current_volatility < volatility_percentile[0]:  # Below 30th percentile
            # In low volatility, narrow thresholds
            adjusted_thresholds = {
                'directional': {
                    'strong': base_thresholds['directional']['strong'] * 0.7,
                    'mild': base_thresholds['directional']['mild'] * 0.8,
                    'neutral': base_thresholds['directional']['neutral'] * 0.9
                },
                'volatility': {
                    'high': base_thresholds['volatility']['high'] * 1.1,  # Raise threshold to catch less high volatility
                    'low': base_thresholds['volatility']['low'] * 1.2
                }
            }
        else:  # Normal volatility
            adjusted_thresholds = base_thresholds
        
        logger.info(f"Adjusted thresholds based on volatility: {adjusted_thresholds}")
        return adjusted_thresholds
    
    def update_performance(self, component, accuracy):
        """
        Update performance history for a component.
        
        Args:
            component (str): Component name
            accuracy (float): Accuracy of the component's prediction
        """
        if component in self.performance_history:
            self.performance_history[component].append(accuracy)
            
            # Keep only the most recent window_size entries
            if len(self.performance_history[component]) > self.window_size:
                self.performance_history[component] = self.performance_history[component][-self.window_size:]
    
    def adjust_weights_by_performance_and_divergence(self, divergence_scores=None):
        """
        Adjust weights based on recent performance and component divergence.
        
        Args:
            divergence_scores (dict, optional): Dictionary of component divergence scores
            
        Returns:
            dict: Adjusted weights
        """
        # First adjust weights by performance
        adjusted_weights = self.adjust_weights_by_performance()
        
        # If divergence scores are provided, further adjust weights
        if divergence_scores:
            # Calculate average divergence for each component
            avg_divergence = {}
            for component, scores in divergence_scores.items():
                if scores:
                    avg_divergence[component] = sum(scores) / len(scores)
                else:
                    avg_divergence[component] = 0.0  # Default if no scores
            
            # Adjust weights based on divergence (lower weight for higher divergence)
            for component in adjusted_weights:
                if component in avg_divergence:
                    # Apply non-linear adjustment to account for critical divergence thresholds
                    if avg_divergence[component] > 0.7:
                        # High divergence - significant weight reduction
                        adjusted_weights[component] *= (1 - avg_divergence[component] * 0.8)
                    elif avg_divergence[component] > 0.3:
                        # Medium divergence - moderate weight reduction
                        adjusted_weights[component] *= (1 - avg_divergence[component] * 0.5)
                    elif avg_divergence[component] > 0.1:
                        # Low divergence - minimal weight reduction
                        adjusted_weights[component] *= (1 - avg_divergence[component] * 0.2)
            
            # Ensure minimum weights
            adjusted_weights = {
                component: max(weight, self.min_weight)
                for component, weight in adjusted_weights.items()
            }
            
            # Normalize weights to ensure they sum to 1
            total_weight = sum(adjusted_weights.values())
            adjusted_weights = {
                component: weight / total_weight
                for component, weight in adjusted_weights.items()
            }
            
            logger.info(f"Adjusted weights by performance and divergence: {adjusted_weights}")
        
        return adjusted_weights
    
    def get_optimal_weights(self, historical_data=None, actual_regimes=None, component_signals=None, divergence_scores=None):
        """
        Get optimal weights using all available methods.
        
        Args:
            historical_data (pd.DataFrame, optional): Historical data
            actual_regimes (pd.Series, optional): Actual market regimes
            component_signals (dict, optional): Dictionary of component signals
            divergence_scores (dict, optional): Dictionary of component divergence scores
            
        Returns:
            dict: Optimal weights
        """
        # Start with current weights
        optimal_weights = self.current_weights.copy()
        
        # If we have historical data, optimize weights
        if historical_data is not None and actual_regimes is not None and component_signals is not None:
            optimized_weights = self.optimize_weights(historical_data, actual_regimes, component_signals)
            
            # Blend optimized weights with current weights
            for component in optimal_weights:
                if component in optimized_weights:
                    optimal_weights[component] = (
                        0.7 * optimized_weights[component] + 
                        0.3 * optimal_weights[component]
                    )
        
        # Adjust weights by performance and divergence
        performance_divergence_weights = self.adjust_weights_by_performance_and_divergence(divergence_scores)
        
        # Blend performance-adjusted weights with optimal weights
        for component in optimal_weights:
            if component in performance_divergence_weights:
                optimal_weights[component] = (
                    0.5 * performance_divergence_weights[component] + 
                    0.5 * optimal_weights[component]
                )
        
        # Update current weights
        self.current_weights = optimal_weights.copy()
        
        return optimal_weights
