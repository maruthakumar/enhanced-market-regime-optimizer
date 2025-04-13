"""
Consolidated Market Regime Classifier Module

This module implements market regime classification using 18 different regimes,
combining Greek sentiment, trending OI with PA, and technical indicators
to determine the current market state.

Features:
- 18 distinct market regime classifications
- Dynamic weighting of indicators
- Support for different timeframes
- Confidence scoring for regime classifications
"""

import pandas as pd
import numpy as np
import logging
from enum import Enum, auto
from datetime import datetime, timedelta

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
    
    # Neutral regimes
    HIGH_VOLATILE_NEUTRAL = auto()
    NORMAL_VOLATILE_NEUTRAL = auto()
    LOW_VOLATILE_NEUTRAL = auto()
    
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
    Consolidated Market Regime Classifier.
    
    This class implements market regime classification using 18 different regimes,
    combining Greek sentiment, trending OI with PA, and technical indicators
    to determine the current market state.
    """
    
    def __init__(self, config=None):
        """
        Initialize Market Regime Classifier.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Indicator weights
        self.indicator_weights = {
            'greek_sentiment': float(self.config.get('greek_sentiment_weight', 0.40)),
            'trending_oi_pa': float(self.config.get('trending_oi_pa_weight', 0.30)),
            'ema': float(self.config.get('ema_weight', 0.10)),
            'vwap': float(self.config.get('vwap_weight', 0.10)),
            'atr': float(self.config.get('atr_weight', 0.05)),
            'volume': float(self.config.get('volume_weight', 0.05))
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
            'mild_bearish': float(self.config.get('mild_bearish_threshold', -0.20)),
            'strong_bearish': float(self.config.get('strong_bearish_threshold', -0.50))
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': float(self.config.get('high_confidence_threshold', 0.75)),
            'medium': float(self.config.get('medium_confidence_threshold', 0.50)),
            'low': float(self.config.get('low_confidence_threshold', 0.25))
        }
        
        logger.info(f"Initialized Market Regime Classifier with {len(REGIME_NAMES)} regimes")
        logger.info(f"Using indicator weights: {self.indicator_weights}")
    
    def classify_regime(self, data_frame, **kwargs):
        """
        Classify market regime.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - greek_sentiment_column (str): Column name for Greek sentiment
                - trending_oi_pa_column (str): Column name for trending OI with PA
                - ema_column (str): Column name for EMA
                - vwap_column (str): Column name for VWAP
                - atr_column (str): Column name for ATR
                - volume_column (str): Column name for volume
                - price_column (str): Column name for price
                - date_column (str): Column name for date
                - time_column (str): Column name for time
            
        Returns:
            pd.DataFrame: Data with classified market regime
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Get column names from kwargs or use defaults
        greek_sentiment_column = kwargs.get('greek_sentiment_column', 'Greek_Sentiment')
        trending_oi_pa_column = kwargs.get('trending_oi_pa_column', 'OI_PA_Regime')
        ema_column = kwargs.get('ema_column', 'EMA_Signal')
        vwap_column = kwargs.get('vwap_column', 'VWAP_Signal')
        atr_column = kwargs.get('atr_column', 'ATR')
        volume_column = kwargs.get('volume_column', 'Volume')
        price_column = kwargs.get('price_column', 'Close')
        date_column = kwargs.get('date_column', 'Date')
        time_column = kwargs.get('time_column', 'Time')
        
        # Check if required columns exist
        required_columns = [price_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return df
        
        # Step 1: Calculate directional component
        directional_component = self._calculate_directional_component(
            df, 
            greek_sentiment_column, 
            trending_oi_pa_column, 
            ema_column, 
            vwap_column
        )
        df['Directional_Component'] = directional_component['value']
        df['Directional_Confidence'] = directional_component['confidence']
        
        # Step 2: Calculate volatility component
        volatility_component = self._calculate_volatility_component(
            df,
            atr_column,
            volume_column,
            price_column
        )
        df['Volatility_Component'] = volatility_component['value']
        df['Volatility_Confidence'] = volatility_component['confidence']
        
        # Step 3: Classify market regime
        regime_classification = self._classify_market_regime(
            df,
            'Directional_Component',
            'Volatility_Component',
            'Directional_Confidence',
            'Volatility_Confidence'
        )
        df['Market_Regime'] = regime_classification['regime']
        df['Market_Regime_Confidence'] = regime_classification['confidence']
        
        # Step 4: Calculate 1-minute rolling market regime
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
        
        logger.info(f"Classified market regime")
        
        return df
    
    def _calculate_directional_component(self, data, greek_sentiment_column, trending_oi_pa_column, ema_column, vwap_column):
        """
        Calculate directional component.
        
        Args:
            data (pd.DataFrame): Input data
            greek_sentiment_column (str): Column name for Greek sentiment
            trending_oi_pa_column (str): Column name for trending OI with PA
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
        
        # Process EMA if available
        if ema_column in data.columns:
            ema_value = data[ema_column].astype(float)
            directional_component += ema_value * self.indicator_weights['ema']
            available_indicators += 1
        
        # Process VWAP if available
        if vwap_column in data.columns:
            vwap_value = data[vwap_column].astype(float)
            directional_component += vwap_value * self.indicator_weights['vwap']
            available_indicators += 1
        
        # Calculate confidence based on available indicators
        for i in range(len(data)):
            if available_indicators.iloc[i] > 0:
                # Normalize directional component by available indicators
                directional_component.iloc[i] = directional_component.iloc[i] / available_indicators.iloc[i]
                
                # Calculate confidence based on number of available indicators
                directional_confidence.iloc[i] = min(1.0, available_indicators.iloc[i] / 4)
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
            elif 'strong_bearish' in sentiment:
                value.iloc[i] = -1.0
            elif 'mild_bearish' in sentiment:
                value.iloc[i] = -0.5
            elif 'neutral' in sentiment:
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
            elif 'strong_bearish' in trend:
                value.iloc[i] = -1.0
            elif 'mild_bearish' in trend:
                value.iloc[i] = -0.5
            elif 'neutral' in trend:
                value.iloc[i] = 0.0
            
            # Add confirmation bonus
            if 'confirmed' in trend:
                if value.iloc[i] > 0:
                    value.iloc[i] = min(1.0, value.iloc[i] + 0.2)
                elif value.iloc[i] < 0:
                    value.iloc[i] = max(-1.0, value.iloc[i] - 0.2)
        
        return value
    
    def _calculate_volatility_component(self, data, atr_column, volume_column, price_column):
        """
        Calculate volatility component.
        
        Args:
            data (pd.DataFrame): Input data
            atr_column (str): Column name for ATR
            volume_column (str): Column name for volume
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
        if atr_column in data.columns:
            # Calculate ATR relative to price
            atr_relative = data[atr_column] / data[price_column]
            
            # Calculate rolling average and standard deviation
            atr_rolling_avg = atr_relative.rolling(window=20, min_periods=1).mean()
            atr_rolling_std = atr_relative.rolling(window=20, min_periods=1).std()
            
            # Calculate z-score
            atr_z_score = (atr_relative - atr_rolling_avg) / atr_rolling_std.replace(0, 1)
            
            # Convert z-score to volatility component
            for i in range(len(data)):
                if pd.notna(atr_z_score.iloc[i]):
                    if atr_z_score.iloc[i] > 1.0:
                        volatility_component.iloc[i] += 1.0
                    elif atr_z_score.iloc[i] < -1.0:
                        volatility_component.iloc[i] += -1.0
                    else:
                        volatility_component.iloc[i] += 0.0
                    
                    available_indicators.iloc[i] += 1
        
        # Process volume if available
        if volume_column in data.columns:
            # Calculate rolling average and standard deviation
            volume_rolling_avg = data[volume_column].rolling(window=20, min_periods=1).mean()
            volume_rolling_std = data[volume_column].rolling(window=20, min_periods=1).std()
            
            # Calculate z-score
            volume_z_score = (data[volume_column] - volume_rolling_avg) / volume_rolling_std.replace(0, 1)
            
            # Convert z-score to volatility component
            for i in range(len(data)):
                if pd.notna(volume_z_score.iloc[i]):
                    if volume_z_score.iloc[i] > 1.0:
                        volatility_component.iloc[i] += 1.0
                    elif volume_z_score.iloc[i] < -1.0:
                        volatility_component.iloc[i] += -1.0
                    else:
                        volatility_component.iloc[i] += 0.0
                    
                    available_indicators.iloc[i] += 1
        
        # Calculate price volatility
        price_returns = data[price_column].pct_change()
        price_volatility = price_returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        # Convert price volatility to volatility component
        for i in range(len(data)):
            if pd.notna(price_volatility.iloc[i]):
                if price_volatility.iloc[i] > self.volatility_thresholds['high']:
                    volatility_component.iloc[i] += 1.0
                elif price_volatility.iloc[i] < self.volatility_thresholds['low']:
                    volatility_component.iloc[i] += -1.0
                else:
                    volatility_component.iloc[i] += 0.0
                
                available_indicators.iloc[i] += 1
        
        # Calculate confidence based on available indicators
        for i in range(len(data)):
            if available_indicators.iloc[i] > 0:
                # Normalize volatility component by available indicators
                volatility_component.iloc[i] = volatility_component.iloc[i] / available_indicators.iloc[i]
                
                # Calculate confidence based on number of available indicators
                volatility_confidence.iloc[i] = min(1.0, available_indicators.iloc[i] / 3)
            else:
                volatility_component.iloc[i] = 0.0
                volatility_confidence.iloc[i] = 0.0
        
        return {
            'value': volatility_component,
            'confidence': volatility_confidence
        }
    
    def _classify_market_regime(self, data, directional_column, volatility_column, directional_confidence_column, volatility_confidence_column):
        """
        Classify market regime.
        
        Args:
            data (pd.DataFrame): Input data
            directional_column (str): Column name for directional component
            volatility_column (str): Column name for volatility component
            directional_confidence_column (str): Column name for directional confidence
            volatility_confidence_column (str): Column name for volatility confidence
            
        Returns:
            dict: Dictionary with regime and confidence
        """
        # Initialize regime and confidence series
        regime = pd.Series("", index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        
        # Classify regime for each row
        for i in range(len(data)):
            # Get values
            directional_value = data[directional_column].iloc[i]
            volatility_value = data[volatility_column].iloc[i]
            directional_confidence = data[directional_confidence_column].iloc[i]
            volatility_confidence = data[volatility_confidence_column].iloc[i]
            
            # Determine directional regime
            if directional_value >= self.directional_thresholds['strong_bullish']:
                directional_regime = "Strong_Bullish"
            elif directional_value >= self.directional_thresholds['mild_bullish']:
                directional_regime = "Mild_Bullish"
            elif directional_value <= self.directional_thresholds['strong_bearish']:
                directional_regime = "Strong_Bearish"
            elif directional_value <= self.directional_thresholds['mild_bearish']:
                directional_regime = "Mild_Bearish"
            else:
                directional_regime = "Neutral"
            
            # Determine volatility regime
            if volatility_value >= self.volatility_thresholds['high']:
                volatility_regime = "High_Volatile"
            elif volatility_value <= -self.volatility_thresholds['low']:
                volatility_regime = "Low_Volatile"
            else:
                volatility_regime = "Normal_Volatile"
            
            # Combine regimes
            combined_regime = f"{volatility_regime}_{directional_regime}"
            
            # Check for transitional regimes
            if i > 0:
                prev_directional_value = data[directional_column].iloc[i-1]
                
                if prev_directional_value >= self.directional_thresholds['mild_bullish'] and directional_value <= self.directional_thresholds['mild_bearish']:
                    combined_regime = REGIME_NAMES[MarketRegime.BULLISH_TO_BEARISH_TRANSITION]
                elif prev_directional_value <= self.directional_thresholds['mild_bearish'] and directional_value >= self.directional_thresholds['mild_bullish']:
                    combined_regime = REGIME_NAMES[MarketRegime.BEARISH_TO_BULLISH_TRANSITION]
                
                prev_volatility_value = data[volatility_column].iloc[i-1]
                
                if abs(volatility_value - prev_volatility_value) > self.volatility_thresholds['high']:
                    combined_regime = REGIME_NAMES[MarketRegime.VOLATILITY_EXPANSION]
            
            # Set regime
            regime.iloc[i] = combined_regime
            
            # Calculate confidence
            confidence.iloc[i] = (directional_confidence + volatility_confidence) / 2
        
        return {
            'regime': regime,
            'confidence': confidence
        }
    
    def _calculate_rolling_market_regime(self, data, regime_column, confidence_column, date_column, time_column):
        """
        Calculate 1-minute rolling market regime.
        
        Args:
            data (pd.DataFrame): Input data
            regime_column (str): Column name for market regime
            confidence_column (str): Column name for market regime confidence
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with regime and confidence
        """
        # Initialize rolling regime and confidence series
        rolling_regime = pd.Series("", index=data.index)
        rolling_confidence = pd.Series(0.0, index=data.index)
        
        # Group by date and time
        grouped = data.groupby([date_column, time_column])
        
        # Calculate rolling regime for each group
        for (date, time), group in grouped:
            # Get indices for this group
            indices = group.index
            
            # Get regimes and confidences for this group
            regimes = data.loc[indices, regime_column]
            confidences = data.loc[indices, confidence_column]
            
            # Count occurrences of each regime
            regime_counts = regimes.value_counts()
            
            if not regime_counts.empty:
                # Get most common regime
                most_common_regime = regime_counts.index[0]
                
                # Calculate average confidence for most common regime
                avg_confidence = confidences[regimes == most_common_regime].mean()
                
                # Set rolling regime and confidence
                rolling_regime.loc[indices] = most_common_regime
                rolling_confidence.loc[indices] = avg_confidence
        
        return {
            'regime': rolling_regime,
            'confidence': rolling_confidence
        }

# Function to classify market regime (for backward compatibility)
def classify_market_regime(market_data, config=None):
    """
    Classify market regime based on market data.
    
    Args:
        market_data (DataFrame): Market data
        config (dict): Configuration settings
        
    Returns:
        Series: Market regime values
    """
    logger.info("Classifying market regime")
    
    try:
        # Create market regime classifier
        classifier = MarketRegimeClassifier(config)
        
        # Classify market regime
        result_df = classifier.classify_regime(market_data)
        
        # Return market regime series
        if 'Market_Regime' in result_df.columns:
            return result_df['Market_Regime']
        else:
            logger.warning("Market_Regime column not found in result")
            return None
    
    except Exception as e:
        logger.error(f"Error classifying market regime: {str(e)}")
        return None
