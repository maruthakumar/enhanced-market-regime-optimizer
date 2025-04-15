"""
Enhanced Market Regime Classifier Module

This module implements market regime classification using 18 different regimes,
combining Greek sentiment, trending OI with PA, and technical indicators
to determine the current market state. It includes dynamic weight adjustment,
expanded regime classifications, and integration with enhanced OI pattern analysis.

Features:
- 18 distinct market regime classifications
- Dynamic weighting of indicators based on historical performance
- Support for different timeframes and DTE-specific analysis
- Expanded regime classifications including sideways patterns
- Integration with enhanced OI pattern analysis
- Confidence scoring for regime classifications
- Early detection of regime transitions
"""

import pandas as pd
import numpy as np
import logging
from enum import Enum, auto
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.metrics import f1_score

# Setup logging
logger = logging.getLogger(__name__)

# Define market regime enum to prevent typos and ensure consistency
class MarketRegime(Enum):
    # Bullish regimes
    HIGH_VOLATILE_STRONG_BULLISH = auto()
    NORMAL_VOLATILE_STRONG_BULLISH = auto()
    LOW_VOLATILE_STRONG_BULLISH = auto()
    HIGH_VOLATILE_MILD_BULLISH = auto()
    NORMAL_VOLATILE_MILD_BULLISH = auto()
    LOW_VOLATILE_MILD_BULLISH = auto()
    
    # Neutral/Sideways regimes
    HIGH_VOLATILE_NEUTRAL = auto()
    NORMAL_VOLATILE_NEUTRAL = auto()
    LOW_VOLATILE_NEUTRAL = auto()
    HIGH_VOLATILE_SIDEWAYS = auto()
    NORMAL_VOLATILE_SIDEWAYS = auto()
    LOW_VOLATILE_SIDEWAYS = auto()
    
    # Sideways-to-Directional regimes
    HIGH_VOLATILE_SIDEWAYS_TO_BULLISH = auto()
    NORMAL_VOLATILE_SIDEWAYS_TO_BULLISH = auto()
    LOW_VOLATILE_SIDEWAYS_TO_BULLISH = auto()
    HIGH_VOLATILE_SIDEWAYS_TO_BEARISH = auto()
    NORMAL_VOLATILE_SIDEWAYS_TO_BEARISH = auto()
    LOW_VOLATILE_SIDEWAYS_TO_BEARISH = auto()
    
    # Bearish regimes
    HIGH_VOLATILE_MILD_BEARISH = auto()
    NORMAL_VOLATILE_MILD_BEARISH = auto()
    LOW_VOLATILE_MILD_BEARISH = auto()
    HIGH_VOLATILE_STRONG_BEARISH = auto()
    NORMAL_VOLATILE_STRONG_BEARISH = auto()
    LOW_VOLATILE_STRONG_BEARISH = auto()
    
    # Transitional regimes
    BULLISH_TO_BEARISH_TRANSITION = auto()
    BEARISH_TO_BULLISH_TRANSITION = auto()
    VOLATILITY_EXPANSION = auto()

# Map enum to string representation
REGIME_NAMES = {
    MarketRegime.HIGH_VOLATILE_STRONG_BULLISH: "High_Volatile_Strong_Bullish",
    MarketRegime.NORMAL_VOLATILE_STRONG_BULLISH: "Normal_Volatile_Strong_Bullish",
    MarketRegime.LOW_VOLATILE_STRONG_BULLISH: "Low_Volatile_Strong_Bullish",
    MarketRegime.HIGH_VOLATILE_MILD_BULLISH: "High_Volatile_Mild_Bullish",
    MarketRegime.NORMAL_VOLATILE_MILD_BULLISH: "Normal_Volatile_Mild_Bullish",
    MarketRegime.LOW_VOLATILE_MILD_BULLISH: "Low_Volatile_Mild_Bullish",
    MarketRegime.HIGH_VOLATILE_NEUTRAL: "High_Volatile_Neutral",
    MarketRegime.NORMAL_VOLATILE_NEUTRAL: "Normal_Volatile_Neutral",
    MarketRegime.LOW_VOLATILE_NEUTRAL: "Low_Volatile_Neutral",
    MarketRegime.HIGH_VOLATILE_SIDEWAYS: "High_Volatile_Sideways",
    MarketRegime.NORMAL_VOLATILE_SIDEWAYS: "Normal_Volatile_Sideways",
    MarketRegime.LOW_VOLATILE_SIDEWAYS: "Low_Volatile_Sideways",
    MarketRegime.HIGH_VOLATILE_SIDEWAYS_TO_BULLISH: "High_Volatile_Sideways_To_Bullish",
    MarketRegime.NORMAL_VOLATILE_SIDEWAYS_TO_BULLISH: "Normal_Volatile_Sideways_To_Bullish",
    MarketRegime.LOW_VOLATILE_SIDEWAYS_TO_BULLISH: "Low_Volatile_Sideways_To_Bullish",
    MarketRegime.HIGH_VOLATILE_SIDEWAYS_TO_BEARISH: "High_Volatile_Sideways_To_Bearish",
    MarketRegime.NORMAL_VOLATILE_SIDEWAYS_TO_BEARISH: "Normal_Volatile_Sideways_To_Bearish",
    MarketRegime.LOW_VOLATILE_SIDEWAYS_TO_BEARISH: "Low_Volatile_Sideways_To_Bearish",
    MarketRegime.HIGH_VOLATILE_MILD_BEARISH: "High_Volatile_Mild_Bearish",
    MarketRegime.NORMAL_VOLATILE_MILD_BEARISH: "Normal_Volatile_Mild_Bearish",
    MarketRegime.LOW_VOLATILE_MILD_BEARISH: "Low_Volatile_Mild_Bearish",
    MarketRegime.HIGH_VOLATILE_STRONG_BEARISH: "High_Volatile_Strong_Bearish",
    MarketRegime.NORMAL_VOLATILE_STRONG_BEARISH: "Normal_Volatile_Strong_Bearish",
    MarketRegime.LOW_VOLATILE_STRONG_BEARISH: "Low_Volatile_Strong_Bearish",
    MarketRegime.BULLISH_TO_BEARISH_TRANSITION: "Bullish_To_Bearish_Transition",
    MarketRegime.BEARISH_TO_BULLISH_TRANSITION: "Bearish_To_Bullish_Transition",
    MarketRegime.VOLATILITY_EXPANSION: "Volatility_Expansion"
}

class MarketRegimeClassifier:
    """
    Enhanced Market Regime Classifier.
    
    This class implements market regime classification using 18 different regimes,
    combining Greek sentiment, trending OI with PA, and technical indicators
    to determine the current market state. It includes dynamic weight adjustment,
    expanded regime classifications, and integration with enhanced OI pattern analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize Market Regime Classifier.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Indicator weights - these will be dynamically adjusted
        self.indicator_weights = {
            'greek_sentiment': float(self.config.get('greek_sentiment_weight', 0.40)),
            'trending_oi_pa': float(self.config.get('trending_oi_pa_weight', 0.30)),
            'iv_skew': float(self.config.get('iv_skew_weight', 0.10)),
            'ema': float(self.config.get('ema_weight', 0.10)),
            'vwap': float(self.config.get('vwap_weight', 0.05)),
            'atr': float(self.config.get('atr_weight', 0.05))
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.indicator_weights.values())
        for key in self.indicator_weights:
            self.indicator_weights[key] /= weight_sum
        
        # Volatility thresholds
        self.volatility_thresholds = {
            'high': float(self.config.get('high_volatility_threshold', 0.20)),
            'low': float(self.config.get('low_volatility_threshold', 0.10))
        }
        
        # Directional thresholds
        self.directional_thresholds = {
            'strong_bullish': float(self.config.get('strong_bullish_threshold', 0.50)),
            'mild_bullish': float(self.config.get('mild_bullish_threshold', 0.20)),
            'sideways_to_bullish': float(self.config.get('sideways_to_bullish_threshold', 0.10)),
            'sideways_to_bearish': float(self.config.get('sideways_to_bearish_threshold', -0.10)),
            'mild_bearish': float(self.config.get('mild_bearish_threshold', -0.20)),
            'strong_bearish': float(self.config.get('strong_bearish_threshold', -0.50))
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': float(self.config.get('high_confidence_threshold', 0.75)),
            'medium': float(self.config.get('medium_confidence_threshold', 0.50)),
            'low': float(self.config.get('low_confidence_threshold', 0.25))
        }
        
        # Dynamic weight adjustment parameters
        self.use_dynamic_weights = bool(self.config.get('use_dynamic_weights', True))
        self.learning_rate = float(self.config.get('learning_rate', 0.1))
        self.window_size = int(self.config.get('window_size', 30))
        
        # Weight history for tracking
        self.weight_history = {key: [value] for key, value in self.indicator_weights.items()}
        
        # Performance metrics history
        self.performance_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Transition detection parameters
        self.transition_lookback = int(self.config.get('transition_lookback', 5))
        self.transition_threshold = float(self.config.get('transition_threshold', 0.3))
        
        # DTE-specific parameters
        self.use_dte_specific = bool(self.config.get('use_dte_specific', True))
        self.dte_groups = {
            'weekly': (0, 7),
            'biweekly': (8, 14),
            'monthly': (15, 30),
            'quarterly': (31, 90)
        }
        
        # Time-of-day adjustment parameters
        self.use_time_of_day_adjustments = bool(self.config.get('use_time_of_day_adjustments', True))
        self.time_of_day_periods = {
            'opening': ('09:15', '10:30'),
            'mid_day': ('10:30', '14:00'),
            'closing': ('14:00', '15:30')
        }
        
        # Time-of-day specific weights
        self.time_of_day_weights = self.config.get('time_of_day_weights', {
            'opening': {
                'greek_sentiment': 0.45,
                'trending_oi_pa': 0.35,
                'iv_skew': 0.10,
                'ema': 0.05,
                'vwap': 0.05
            },
            'mid_day': {
                'greek_sentiment': 0.40,
                'trending_oi_pa': 0.30,
                'iv_skew': 0.10,
                'ema': 0.10,
                'vwap': 0.10
            },
            'closing': {
                'greek_sentiment': 0.35,
                'trending_oi_pa': 0.35,
                'iv_skew': 0.15,
                'ema': 0.10,
                'vwap': 0.05
            }
        })
        
        # Multi-timeframe analysis parameters
        self.use_multi_timeframe = bool(self.config.get('use_multi_timeframe', True))
        self.timeframes = self.config.get('timeframes', ['5min', '15min', '1hour'])
        self.timeframe_weights = self.config.get('timeframe_weights', {
            '5min': 0.3,
            '15min': 0.3,
            '1hour': 0.4
        })
        
        logger.info(f"Initialized Enhanced Market Regime Classifier with {len(REGIME_NAMES)} regimes")
        logger.info(f"Using indicator weights: {self.indicator_weights}")
        logger.info(f"Dynamic weight adjustment: {self.use_dynamic_weights}")
        logger.info(f"Time-of-day adjustments: {self.use_time_of_day_adjustments}")
        logger.info(f"Multi-timeframe analysis: {self.use_multi_timeframe}")
    
    def classify_regime(self, data_frame, **kwargs):
        """
        Classify market regime.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - greek_sentiment_column (str): Column name for Greek sentiment
                - trending_oi_pa_column (str): Column name for trending OI with PA
                - iv_skew_column (str): Column name for IV skew
                - ema_column (str): Column name for EMA
                - vwap_column (str): Column name for VWAP
                - atr_column (str): Column name for ATR
                - price_column (str): Column name for price
                - date_column (str): Column name for date
                - time_column (str): Column name for time
                - dte_column (str): Column name for DTE
                - specific_dte (int): Specific DTE to use for calculations
                - timeframe (str): Timeframe of the data
            
        Returns:
            pd.DataFrame: Data with classified market regime
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Get column names from kwargs or use defaults
        greek_sentiment_column = kwargs.get('greek_sentiment_column', 'Greek_Sentiment')
        trending_oi_pa_column = kwargs.get('trending_oi_pa_column', 'OI_PA_Regime')
        iv_skew_column = kwargs.get('iv_skew_column', 'IV_Skew')
        ema_column = kwargs.get('ema_column', 'EMA_Signal')
        vwap_column = kwargs.get('vwap_column', 'VWAP_Signal')
        atr_column = kwargs.get('atr_column', 'ATR')
        price_column = kwargs.get('price_column', 'Close')
        date_column = kwargs.get('date_column', 'Date')
        time_column = kwargs.get('time_column', 'Time')
        dte_column = kwargs.get('dte_column', 'DTE')
        specific_dte = kwargs.get('specific_dte', None)
        timeframe = kwargs.get('timeframe', '5min')
        
        # Check if required columns exist
        required_columns = [price_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return df
        
        # Filter by DTE if specified
        if specific_dte is not None and dte_column in df.columns:
            df = df[df[dte_column] == specific_dte].copy()
            logger.info(f"Filtered data for DTE = {specific_dte}")
        
        # Apply time-of-day adjustments if enabled
        if self.use_time_of_day_adjustments and time_column in df.columns:
            df = self._apply_time_of_day_adjustments(df, time_column)
        
        # Step 1: Calculate directional component
        directional_component = self._calculate_directional_component(
            df, 
            greek_sentiment_column, 
            trending_oi_pa_column, 
            iv_skew_column,
            ema_column, 
            vwap_column
        )
        df['Directional_Component'] = directional_component['value']
        df['Directional_Confidence'] = directional_component['confidence']
        
        # Step 2: Calculate volatility component
        volatility_component = self._calculate_volatility_component(
            df,
            atr_column,
            iv_skew_column,
            price_column
        )
        df['Volatility_Component'] = volatility_component['value']
        df['Volatility_Confidence'] = volatility_component['confidence']
        
        # Step 3: Detect transitions
        transitions = self._detect_transitions(
            df,
            'Directional_Component',
            'Volatility_Component',
            date_column,
            time_column
        )
        df['Transition_Type'] = transitions['type']
        df['Transition_Probability'] = transitions['probability']
        
        # Step 4: Classify market regime
        regime_classification = self._classify_market_regime(
            df,
            'Directional_Component',
            'Volatility_Component',
            'Directional_Confidence',
            'Volatility_Confidence',
            'Transition_Type',
            'Transition_Probability'
        )
        df['Market_Regime'] = regime_classification['regime']
        df['Market_Regime_Confidence'] = regime_classification['confidence']
        
        # Step 5: Calculate rolling market regime if date and time columns are available
        if date_column in df.columns and time_column in df.columns:
            rolling_regime = self._calculate_rolling_market_regime(
                df,
                'Market_Regime',
                'Market_Regime_Confidence',
                date_column,
                time_column
            )
            df['Rolling_Market_Regime'] = rolling_regime['regime']
            df['Rolling_Market_Regime_Confidence'] = rolling_regime['confidence']
        
        # Step 6: Update weights if dynamic adjustment is enabled and we have actual regime data
        if self.use_dynamic_weights and 'actual_regime' in df.columns:
            self._update_weights(df)
        
        # Store timeframe information
        df['Timeframe'] = timeframe
        
        logger.info(f"Classified market regime for timeframe {timeframe}")
        
        return df
    
    def _apply_time_of_day_adjustments(self, data, time_column):
        """
        Apply time-of-day adjustments to weights.
        
        Args:
            data (pd.DataFrame): Input data
            time_column (str): Column name for time
            
        Returns:
            pd.DataFrame: Data with time-of-day adjusted weights
        """
        # Make a copy
        df = data.copy()
        
        # Add time period column
        df['Time_Period'] = 'mid_day'  # Default
        
        # Convert time to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            try:
                df['time_temp'] = pd.to_datetime(df[time_column]).dt.time
            except:
                logger.warning(f"Failed to convert {time_column} to time format")
                return df
        else:
            df['time_temp'] = df[time_column].dt.time
        
        # Assign time periods
        for period, (start, end) in self.time_of_day_periods.items():
            start_time = datetime.strptime(start, '%H:%M').time()
            end_time = datetime.strptime(end, '%H:%M').time()
            
            # Assign period if time is between start and end
            mask = (df['time_temp'] >= start_time) & (df['time_temp'] < end_time)
            df.loc[mask, 'Time_Period'] = period
        
        # Drop temporary column
        df.drop('time_temp', axis=1, inplace=True)
        
        # Adjust weights based on time period
        for i, row in df.iterrows():
            period = row['Time_Period']
            if period in self.time_of_day_weights:
                # Store original weights
                df.loc[i, 'Original_Weights'] = str(self.indicator_weights)
                
                # Temporarily adjust weights for this row
                for indicator, weight in self.time_of_day_weights[period].items():
                    if indicator in self.indicator_weights:
                        self.indicator_weights[indicator] = weight
                
                # Normalize weights
                weight_sum = sum(self.indicator_weights.values())
                for key in self.indicator_weights:
                    self.indicator_weights[key] /= weight_sum
                
                # Store adjusted weights
                df.loc[i, 'Adjusted_Weights'] = str(self.indicator_weights)
        
        return df
    
    def _calculate_directional_component(self, data, greek_sentiment_column, trending_oi_pa_column, iv_skew_column, ema_column, vwap_column):
        """
        Calculate directional component.
        
        Args:
            data (pd.DataFrame): Input data
            greek_sentiment_column (str): Column name for Greek sentiment
            trending_oi_pa_column (str): Column name for trending OI with PA
            iv_skew_column (str): Column name for IV skew
            ema_column (str): Column name for EMA
            vwap_column (str): Column name for VWAP
            
        Returns:
            dict: Dictionary with value and confidence
        """
        # Initialize directional component and confidence
        directional_component = pd.Series(0.0, index=data.index)
        directional_confidence = pd.Series(0.0, index=data.index)
        
        # Track available indicators for each row
        available_indicators = pd.Series(0, index=data.index)
        
        # Process Greek sentiment if available
        if greek_sentiment_column in data.columns:
            greek_sentiment_value = self._convert_greek_sentiment_to_value(data[greek_sentiment_column])
            directional_component += greek_sentiment_value * self.indicator_weights['greek_sentiment']
            available_indicators += 1
        
        # Process trending OI with PA if available
        if trending_oi_pa_column in data.columns:
            trending_oi_pa_value = self._convert_trending_oi_pa_to_value(data[trending_oi_pa_column])
            directional_component += trending_oi_pa_value * self.indicator_weights['trending_oi_pa']
            available_indicators += 1
        
        # Process IV skew if available
        if iv_skew_column in data.columns:
            iv_skew_value = self._convert_iv_skew_to_value(data[iv_skew_column])
            directional_component += iv_skew_value * self.indicator_weights['iv_skew']
            available_indicators += 1
        
        # Process EMA if available
        if ema_column in data.columns:
            ema_value = self._convert_ema_to_value(data[ema_column])
            directional_component += ema_value * self.indicator_weights['ema']
            available_indicators += 1
        
        # Process VWAP if available
        if vwap_column in data.columns:
            vwap_value = self._convert_vwap_to_value(data[vwap_column])
            directional_component += vwap_value * self.indicator_weights['vwap']
            available_indicators += 1
        
        # Calculate confidence based on available indicators
        for i in range(len(data)):
            if available_indicators.iloc[i] > 0:
                # Normalize directional component by available indicators
                directional_component.iloc[i] = directional_component.iloc[i] / sum(
                    weight for indicator, weight in self.indicator_weights.items() 
                    if indicator in ['greek_sentiment', 'trending_oi_pa', 'iv_skew', 'ema', 'vwap']
                )
                
                # Calculate confidence based on number of available indicators
                directional_confidence.iloc[i] = min(1.0, available_indicators.iloc[i] / 5)
            else:
                directional_component.iloc[i] = 0.0
                directional_confidence.iloc[i] = 0.0
        
        return {
            'value': directional_component,
            'confidence': directional_confidence
        }
    
    def _convert_greek_sentiment_to_value(self, greek_sentiment):
        """
        Convert Greek sentiment to numerical value.
        
        Args:
            greek_sentiment (pd.Series): Greek sentiment
            
        Returns:
            pd.Series: Numerical value
        """
        # Initialize value series
        value = pd.Series(0.0, index=greek_sentiment.index)
        
        # Convert each sentiment to value
        for i in range(len(greek_sentiment)):
            sentiment = str(greek_sentiment.iloc[i]).lower()
            
            if 'strong_bullish' in sentiment:
                value.iloc[i] = 1.0
            elif 'mild_bullish' in sentiment:
                value.iloc[i] = 0.5
            elif 'sideways_to_bullish' in sentiment:
                value.iloc[i] = 0.2
            elif 'sideways_to_bearish' in sentiment:
                value.iloc[i] = -0.2
            elif 'strong_bearish' in sentiment:
                value.iloc[i] = -1.0
            elif 'mild_bearish' in sentiment:
                value.iloc[i] = -0.5
            elif 'neutral' in sentiment or 'sideways' in sentiment:
                value.iloc[i] = 0.0
            
            # Add confirmation bonus
            if 'confirmed' in sentiment:
                if value.iloc[i] > 0:
                    value.iloc[i] = min(1.0, value.iloc[i] + 0.2)
                elif value.iloc[i] < 0:
                    value.iloc[i] = max(-1.0, value.iloc[i] - 0.2)
        
        return value
    
    def _convert_trending_oi_pa_to_value(self, trending_oi_pa):
        """
        Convert trending OI with PA to numerical value.
        
        Args:
            trending_oi_pa (pd.Series): Trending OI with PA
            
        Returns:
            pd.Series: Numerical value
        """
        # Initialize value series
        value = pd.Series(0.0, index=trending_oi_pa.index)
        
        # Convert each trending OI with PA to value
        for i in range(len(trending_oi_pa)):
            trend = str(trending_oi_pa.iloc[i]).lower()
            
            if 'strong_bullish' in trend:
                value.iloc[i] = 1.0
            elif 'mild_bullish' in trend:
                value.iloc[i] = 0.5
            elif 'long_build_up' in trend:
                value.iloc[i] = 0.7
            elif 'short_covering' in trend:
                value.iloc[i] = 0.6
            elif 'strong_bearish' in trend:
                value.iloc[i] = -1.0
            elif 'mild_bearish' in trend:
                value.iloc[i] = -0.5
            elif 'short_build_up' in trend:
                value.iloc[i] = -0.7
            elif 'long_unwinding' in trend:
                value.iloc[i] = -0.6
            elif 'neutral' in trend:
                value.iloc[i] = 0.0
            elif 'sideways_to_bullish' in trend:
                value.iloc[i] = 0.2
            elif 'sideways_to_bearish' in trend:
                value.iloc[i] = -0.2
            elif 'sideways' in trend:
                value.iloc[i] = 0.0
        
        return value
    
    def _convert_iv_skew_to_value(self, iv_skew):
        """
        Convert IV skew to numerical value.
        
        Args:
            iv_skew (pd.Series): IV skew
            
        Returns:
            pd.Series: Numerical value
        """
        # Initialize value series
        value = pd.Series(0.0, index=iv_skew.index)
        
        # If IV skew is already numerical, normalize it to [-1, 1]
        if pd.api.types.is_numeric_dtype(iv_skew):
            # Calculate min and max for normalization
            min_val = iv_skew.min()
            max_val = iv_skew.max()
            
            if max_val > min_val:
                # Normalize to [-1, 1]
                value = 2 * (iv_skew - min_val) / (max_val - min_val) - 1
            
            # Invert the value since positive skew is bearish and negative skew is bullish
            value = -value
        else:
            # Convert each IV skew to value
            for i in range(len(iv_skew)):
                skew = str(iv_skew.iloc[i]).lower()
                
                if 'bullish' in skew:
                    value.iloc[i] = 0.5
                elif 'bearish' in skew:
                    value.iloc[i] = -0.5
                elif 'neutral' in skew:
                    value.iloc[i] = 0.0
                
                # Add strength modifier
                if 'strong' in skew:
                    value.iloc[i] *= 2.0
                elif 'mild' in skew:
                    value.iloc[i] *= 0.5
        
        return value
    
    def _convert_ema_to_value(self, ema):
        """
        Convert EMA to numerical value.
        
        Args:
            ema (pd.Series): EMA
            
        Returns:
            pd.Series: Numerical value
        """
        # Initialize value series
        value = pd.Series(0.0, index=ema.index)
        
        # If EMA is already numerical, normalize it to [-1, 1]
        if pd.api.types.is_numeric_dtype(ema):
            # Calculate min and max for normalization
            min_val = ema.min()
            max_val = ema.max()
            
            if max_val > min_val:
                # Normalize to [-1, 1]
                value = 2 * (ema - min_val) / (max_val - min_val) - 1
        else:
            # Convert each EMA to value
            for i in range(len(ema)):
                ema_val = str(ema.iloc[i]).lower()
                
                if 'bullish' in ema_val:
                    value.iloc[i] = 0.5
                elif 'bearish' in ema_val:
                    value.iloc[i] = -0.5
                elif 'neutral' in ema_val:
                    value.iloc[i] = 0.0
                
                # Add strength modifier
                if 'strong' in ema_val:
                    value.iloc[i] *= 2.0
                elif 'mild' in ema_val:
                    value.iloc[i] *= 0.5
        
        return value
    
    def _convert_vwap_to_value(self, vwap):
        """
        Convert VWAP to numerical value.
        
        Args:
            vwap (pd.Series): VWAP
            
        Returns:
            pd.Series: Numerical value
        """
        # Initialize value series
        value = pd.Series(0.0, index=vwap.index)
        
        # If VWAP is already numerical, normalize it to [-1, 1]
        if pd.api.types.is_numeric_dtype(vwap):
            # Calculate min and max for normalization
            min_val = vwap.min()
            max_val = vwap.max()
            
            if max_val > min_val:
                # Normalize to [-1, 1]
                value = 2 * (vwap - min_val) / (max_val - min_val) - 1
        else:
            # Convert each VWAP to value
            for i in range(len(vwap)):
                vwap_val = str(vwap.iloc[i]).lower()
                
                if 'above' in vwap_val:
                    value.iloc[i] = 0.5
                elif 'below' in vwap_val:
                    value.iloc[i] = -0.5
                elif 'at' in vwap_val:
                    value.iloc[i] = 0.0
                
                # Add strength modifier
                if 'far' in vwap_val:
                    value.iloc[i] *= 2.0
                elif 'near' in vwap_val:
                    value.iloc[i] *= 0.5
        
        return value
    
    def _calculate_volatility_component(self, data, atr_column, iv_skew_column, price_column):
        """
        Calculate volatility component.
        
        Args:
            data (pd.DataFrame): Input data
            atr_column (str): Column name for ATR
            iv_skew_column (str): Column name for IV skew
            price_column (str): Column name for price
            
        Returns:
            dict: Dictionary with value and confidence
        """
        # Initialize volatility component and confidence
        volatility_component = pd.Series(0.0, index=data.index)
        volatility_confidence = pd.Series(0.0, index=data.index)
        
        # Track available indicators for each row
        available_indicators = pd.Series(0, index=data.index)
        
        # Process ATR if available
        if atr_column in data.columns and price_column in data.columns:
            # Calculate ATR percentage
            atr_percentage = data[atr_column] / data[price_column]
            
            # Normalize to [0, 1]
            max_atr = atr_percentage.max()
            min_atr = atr_percentage.min()
            
            if max_atr > min_atr:
                normalized_atr = (atr_percentage - min_atr) / (max_atr - min_atr)
                volatility_component += normalized_atr
                available_indicators += 1
        
        # Process IV skew if available
        if iv_skew_column in data.columns:
            # Extract volatility component from IV skew
            iv_volatility = self._extract_volatility_from_iv_skew(data[iv_skew_column])
            volatility_component += iv_volatility
            available_indicators += 1
        
        # Calculate confidence based on available indicators
        for i in range(len(data)):
            if available_indicators.iloc[i] > 0:
                # Normalize volatility component by available indicators
                volatility_component.iloc[i] = volatility_component.iloc[i] / available_indicators.iloc[i]
                
                # Calculate confidence based on number of available indicators
                volatility_confidence.iloc[i] = min(1.0, available_indicators.iloc[i] / 2)
            else:
                volatility_component.iloc[i] = 0.5  # Default to medium volatility
                volatility_confidence.iloc[i] = 0.0
        
        return {
            'value': volatility_component,
            'confidence': volatility_confidence
        }
    
    def _extract_volatility_from_iv_skew(self, iv_skew):
        """
        Extract volatility component from IV skew.
        
        Args:
            iv_skew (pd.Series): IV skew
            
        Returns:
            pd.Series: Volatility component
        """
        # Initialize volatility series
        volatility = pd.Series(0.5, index=iv_skew.index)  # Default to medium volatility
        
        # If IV skew is already numerical, use it directly
        if pd.api.types.is_numeric_dtype(iv_skew):
            # Calculate absolute value of IV skew
            abs_skew = iv_skew.abs()
            
            # Normalize to [0, 1]
            max_skew = abs_skew.max()
            min_skew = abs_skew.min()
            
            if max_skew > min_skew:
                volatility = (abs_skew - min_skew) / (max_skew - min_skew)
        else:
            # Extract volatility from string representation
            for i in range(len(iv_skew)):
                skew = str(iv_skew.iloc[i]).lower()
                
                if 'high' in skew or 'strong' in skew:
                    volatility.iloc[i] = 0.8
                elif 'low' in skew or 'mild' in skew:
                    volatility.iloc[i] = 0.2
                else:
                    volatility.iloc[i] = 0.5
        
        return volatility
    
    def _detect_transitions(self, data, directional_column, volatility_column, date_column, time_column):
        """
        Detect transitions in market regime.
        
        Args:
            data (pd.DataFrame): Input data
            directional_column (str): Column name for directional component
            volatility_column (str): Column name for volatility component
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with transition type and probability
        """
        # Initialize transition type and probability
        transition_type = pd.Series('None', index=data.index)
        transition_probability = pd.Series(0.0, index=data.index)
        
        # Check if we have enough data
        if len(data) <= self.transition_lookback:
            return {
                'type': transition_type,
                'probability': transition_probability
            }
        
        # Check if we have date and time columns
        if date_column not in data.columns or time_column not in data.columns:
            return {
                'type': transition_type,
                'probability': transition_probability
            }
        
        # Sort data by date and time
        data_sorted = data.sort_values([date_column, time_column])
        
        # Calculate directional change
        directional_change = data_sorted[directional_column].diff()
        
        # Calculate volatility change
        volatility_change = data_sorted[volatility_column].diff()
        
        # Detect transitions
        for i in range(self.transition_lookback, len(data_sorted)):
            # Get current index
            current_index = data_sorted.index[i]
            
            # Calculate average directional change over lookback period
            avg_directional_change = directional_change.iloc[i-self.transition_lookback:i].mean()
            
            # Calculate average volatility change over lookback period
            avg_volatility_change = volatility_change.iloc[i-self.transition_lookback:i].mean()
            
            # Detect directional transitions
            if avg_directional_change > self.transition_threshold:
                transition_type.loc[current_index] = 'Bearish_To_Bullish'
                transition_probability.loc[current_index] = min(1.0, abs(avg_directional_change) / (2 * self.transition_threshold))
            elif avg_directional_change < -self.transition_threshold:
                transition_type.loc[current_index] = 'Bullish_To_Bearish'
                transition_probability.loc[current_index] = min(1.0, abs(avg_directional_change) / (2 * self.transition_threshold))
            
            # Detect volatility transitions
            if avg_volatility_change > self.transition_threshold:
                transition_type.loc[current_index] = 'Volatility_Expansion'
                transition_probability.loc[current_index] = min(1.0, abs(avg_volatility_change) / (2 * self.transition_threshold))
        
        return {
            'type': transition_type,
            'probability': transition_probability
        }
    
    def _classify_market_regime(self, data, directional_column, volatility_column, directional_confidence_column, volatility_confidence_column, transition_type_column, transition_probability_column):
        """
        Classify market regime.
        
        Args:
            data (pd.DataFrame): Input data
            directional_column (str): Column name for directional component
            volatility_column (str): Column name for volatility component
            directional_confidence_column (str): Column name for directional confidence
            volatility_confidence_column (str): Column name for volatility confidence
            transition_type_column (str): Column name for transition type
            transition_probability_column (str): Column name for transition probability
            
        Returns:
            dict: Dictionary with regime and confidence
        """
        # Initialize regime and confidence
        regime = pd.Series('Unknown', index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        
        # Classify each row
        for i, row in data.iterrows():
            # Get directional component
            directional = row[directional_column]
            
            # Get volatility component
            volatility = row[volatility_column]
            
            # Get confidence values
            directional_confidence = row[directional_confidence_column]
            volatility_confidence = row[volatility_confidence_column]
            
            # Get transition information
            transition_type = row[transition_type_column]
            transition_probability = row[transition_probability_column]
            
            # Check for transitions first
            if transition_type != 'None' and transition_probability > 0.5:
                if transition_type == 'Bearish_To_Bullish':
                    regime.loc[i] = REGIME_NAMES[MarketRegime.BEARISH_TO_BULLISH_TRANSITION]
                    confidence.loc[i] = transition_probability
                elif transition_type == 'Bullish_To_Bearish':
                    regime.loc[i] = REGIME_NAMES[MarketRegime.BULLISH_TO_BEARISH_TRANSITION]
                    confidence.loc[i] = transition_probability
                elif transition_type == 'Volatility_Expansion':
                    regime.loc[i] = REGIME_NAMES[MarketRegime.VOLATILITY_EXPANSION]
                    confidence.loc[i] = transition_probability
                continue
            
            # Determine volatility level
            if volatility >= self.volatility_thresholds['high']:
                volatility_level = 'HIGH'
            elif volatility <= self.volatility_thresholds['low']:
                volatility_level = 'LOW'
            else:
                volatility_level = 'NORMAL'
            
            # Determine directional bias
            if directional >= self.directional_thresholds['strong_bullish']:
                directional_bias = 'STRONG_BULLISH'
            elif directional >= self.directional_thresholds['mild_bullish']:
                directional_bias = 'MILD_BULLISH'
            elif directional >= self.directional_thresholds['sideways_to_bullish']:
                directional_bias = 'SIDEWAYS_TO_BULLISH'
            elif directional <= self.directional_thresholds['strong_bearish']:
                directional_bias = 'STRONG_BEARISH'
            elif directional <= self.directional_thresholds['mild_bearish']:
                directional_bias = 'MILD_BEARISH'
            elif directional <= self.directional_thresholds['sideways_to_bearish']:
                directional_bias = 'SIDEWAYS_TO_BEARISH'
            else:
                # Check if it's neutral or sideways
                if abs(directional) < 0.05:
                    directional_bias = 'NEUTRAL'
                else:
                    directional_bias = 'SIDEWAYS'
            
            # Map to regime enum
            regime_key = f"{volatility_level}_{directional_bias}"
            
            # Find matching regime
            matching_regime = None
            for r in MarketRegime:
                if REGIME_NAMES[r].upper() == regime_key:
                    matching_regime = r
                    break
            
            # Assign regime
            if matching_regime:
                regime.loc[i] = REGIME_NAMES[matching_regime]
            else:
                # Default to neutral if no match
                regime.loc[i] = REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_NEUTRAL]
            
            # Calculate confidence
            confidence.loc[i] = (directional_confidence + volatility_confidence) / 2
        
        return {
            'regime': regime,
            'confidence': confidence
        }
    
    def _calculate_rolling_market_regime(self, data, regime_column, confidence_column, date_column, time_column):
        """
        Calculate rolling market regime.
        
        Args:
            data (pd.DataFrame): Input data
            regime_column (str): Column name for market regime
            confidence_column (str): Column name for confidence
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with regime and confidence
        """
        # Initialize rolling regime and confidence
        rolling_regime = data[regime_column].copy()
        rolling_confidence = data[confidence_column].copy()
        
        # Check if we have enough data
        if len(data) <= 1:
            return {
                'regime': rolling_regime,
                'confidence': rolling_confidence
            }
        
        # Check if we have date and time columns
        if date_column not in data.columns or time_column not in data.columns:
            return {
                'regime': rolling_regime,
                'confidence': rolling_confidence
            }
        
        # Sort data by date and time
        data_sorted = data.sort_values([date_column, time_column])
        
        # Calculate rolling regime
        for i in range(1, len(data_sorted)):
            # Get current index
            current_index = data_sorted.index[i]
            
            # Get previous index
            previous_index = data_sorted.index[i-1]
            
            # Get current regime and confidence
            current_regime = data_sorted[regime_column].loc[current_index]
            current_confidence = data_sorted[confidence_column].loc[current_index]
            
            # Get previous rolling regime and confidence
            previous_rolling_regime = rolling_regime.loc[previous_index]
            previous_rolling_confidence = rolling_confidence.loc[previous_index]
            
            # Check if current regime is the same as previous rolling regime
            if current_regime == previous_rolling_regime:
                # Increase confidence
                rolling_confidence.loc[current_index] = min(1.0, current_confidence + 0.1)
            else:
                # Check if current confidence is higher than previous rolling confidence
                if current_confidence > previous_rolling_confidence:
                    # Switch to current regime
                    rolling_regime.loc[current_index] = current_regime
                    rolling_confidence.loc[current_index] = current_confidence
                else:
                    # Keep previous rolling regime but decrease confidence
                    rolling_regime.loc[current_index] = previous_rolling_regime
                    rolling_confidence.loc[current_index] = max(0.0, previous_rolling_confidence - 0.1)
        
        return {
            'regime': rolling_regime,
            'confidence': rolling_confidence
        }
    
    def _update_weights(self, data):
        """
        Update weights based on performance.
        
        Args:
            data (pd.DataFrame): Input data with actual regime
        """
        # Check if we have actual regime
        if 'actual_regime' not in data.columns or 'Market_Regime' not in data.columns:
            logger.warning("Missing actual_regime or Market_Regime column, cannot update weights")
            return
        
        # Calculate performance metrics
        accuracy = (data['Market_Regime'] == data['actual_regime']).mean()
        
        # Convert regimes to numerical values for f1 score
        regime_mapping = {regime: i for i, regime in enumerate(data['Market_Regime'].unique())}
        y_true = data['actual_regime'].map(regime_mapping)
        y_pred = data['Market_Regime'].map(regime_mapping)
        
        # Calculate f1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Update performance history
        self.performance_history['accuracy'].append(accuracy)
        self.performance_history['f1'].append(f1)
        
        # Limit history to window size
        if len(self.performance_history['accuracy']) > self.window_size:
            self.performance_history['accuracy'] = self.performance_history['accuracy'][-self.window_size:]
            self.performance_history['f1'] = self.performance_history['f1'][-self.window_size:]
        
        # Calculate performance change
        if len(self.performance_history['accuracy']) > 1:
            accuracy_change = self.performance_history['accuracy'][-1] - self.performance_history['accuracy'][-2]
            f1_change = self.performance_history['f1'][-1] - self.performance_history['f1'][-2]
            
            # Update weights based on performance change
            for key in self.indicator_weights:
                # Get contribution of this indicator
                contribution = self._calculate_indicator_contribution(data, key)
                
                # Update weight
                weight_change = self.learning_rate * (accuracy_change + f1_change) * contribution
                self.indicator_weights[key] += weight_change
            
            # Normalize weights to sum to 1
            weight_sum = sum(self.indicator_weights.values())
            for key in self.indicator_weights:
                self.indicator_weights[key] /= weight_sum
            
            # Update weight history
            for key, value in self.indicator_weights.items():
                self.weight_history[key].append(value)
                
                # Limit history to window size
                if len(self.weight_history[key]) > self.window_size:
                    self.weight_history[key] = self.weight_history[key][-self.window_size:]
    
    def _calculate_indicator_contribution(self, data, indicator):
        """
        Calculate indicator contribution to performance.
        
        Args:
            data (pd.DataFrame): Input data
            indicator (str): Indicator name
            
        Returns:
            float: Contribution
        """
        # Initialize contribution
        contribution = 0.0
        
        # Check if we have the necessary columns
        if f"{indicator}_value" not in data.columns or 'actual_regime' not in data.columns:
            return contribution
        
        # Calculate correlation between indicator value and actual regime
        indicator_values = data[f"{indicator}_value"]
        
        # Convert actual regime to numerical values
        regime_mapping = {
            'Strong_Bullish': 1.0,
            'Mild_Bullish': 0.5,
            'Sideways_To_Bullish': 0.2,
            'Neutral': 0.0,
            'Sideways': 0.0,
            'Sideways_To_Bearish': -0.2,
            'Mild_Bearish': -0.5,
            'Strong_Bearish': -1.0
        }
        
        actual_values = data['actual_regime'].map(lambda x: regime_mapping.get(x, 0.0))
        
        # Calculate correlation
        correlation = np.corrcoef(indicator_values, actual_values)[0, 1] if len(indicator_values) > 1 else 0.0
        
        # Set contribution based on correlation
        contribution = correlation if not np.isnan(correlation) else 0.0
        
        return contribution
    
    def combine_timeframe_results(self, results):
        """
        Combine results from multiple timeframes.
        
        Args:
            results (list): List of dictionaries with timeframe, result, and weight
            
        Returns:
            dict: Combined result
        """
        # Check if we have results
        if not results:
            logger.warning("No results to combine")
            return None
        
        # Initialize combined result
        combined_result = {
            'Market_Regime': None,
            'Market_Regime_Confidence': 0.0,
            'Timeframes': [],
            'Weights': []
        }
        
        # Initialize regime counts
        regime_counts = {}
        
        # Process each result
        for result_dict in results:
            # Get timeframe, result, and weight
            timeframe = result_dict['timeframe']
            result = result_dict['result']
            weight = result_dict['weight']
            
            # Skip if result is None
            if result is None:
                continue
            
            # Get regime and confidence
            regime = result['Market_Regime'].iloc[-1] if 'Market_Regime' in result.columns else 'Unknown'
            confidence = result['Market_Regime_Confidence'].iloc[-1] if 'Market_Regime_Confidence' in result.columns else 0.0
            
            # Update regime counts
            if regime not in regime_counts:
                regime_counts[regime] = 0.0
            
            regime_counts[regime] += weight * confidence
            
            # Add to combined result
            combined_result['Timeframes'].append(timeframe)
            combined_result['Weights'].append(weight)
        
        # Find regime with highest count
        if regime_counts:
            max_regime = max(regime_counts, key=regime_counts.get)
            max_count = regime_counts[max_regime]
            
            combined_result['Market_Regime'] = max_regime
            combined_result['Market_Regime_Confidence'] = max_count / sum(combined_result['Weights'])
        
        return combined_result
    
    def get_weight_history(self):
        """
        Get weight history.
        
        Returns:
            dict: Weight history
        """
        return self.weight_history
    
    def get_performance_history(self):
        """
        Get performance history.
        
        Returns:
            dict: Performance history
        """
        return self.performance_history
