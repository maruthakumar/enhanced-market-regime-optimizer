"""
Unified Market Regime Pipeline

This module implements a unified pipeline that integrates all components
of the market regime identification system, connecting with the existing
consolidator and optimizer components.

Features:
- Complete market regime identification pipeline
- Integration with consolidator
- Connection to dimensional optimizer
- Multi-timeframe analysis
- Time-of-day adjustments
- Comprehensive logging and error handling
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback

# Import feature engineering components
sys.path.append('/home/ubuntu/market_regime_testing/enhanced-market-regime-optimizer')
from utils.feature_engineering.greek_sentiment.greek_sentiment_analysis import GreekSentimentAnalysis
from utils.feature_engineering.trending_oi_pa.trending_oi_pa_analysis import TrendingOIWithPAAnalysis
from utils.feature_engineering.iv_skew.iv_skew_analysis import IVSkewAnalysis
from utils.feature_engineering.ema_indicators.ema_indicators import EMAIndicators
from utils.feature_engineering.vwap_indicators.vwap_indicators import VWAPIndicators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_regime_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketRegimePipeline:
    """
    Unified Market Regime Pipeline.
    
    This class implements a unified pipeline that integrates all components
    of the market regime identification system.
    """
    
    def __init__(self, config=None):
        """
        Initialize Market Regime Pipeline.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Output directories
        self.output_dir = self.config.get('output_dir', 'output')
        self.visualization_dir = os.path.join(self.output_dir, 'visualizations')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Market regime parameters
        self.market_regimes = [
            'Strong_Bullish',
            'Bullish',
            'Mild_Bullish',
            'Sideways_To_Bullish',
            'Neutral',
            'Sideways',
            'Sideways_To_Bearish',
            'Mild_Bearish',
            'Bearish',
            'Strong_Bearish',
            'Reversal_Imminent_Bullish',
            'Reversal_Imminent_Bearish',
            'Exhaustion_Bullish',
            'Exhaustion_Bearish',
            'Failed_Breakout_Bullish',
            'Failed_Breakout_Bearish',
            'Institutional_Accumulation',
            'Institutional_Distribution'
        ]
        
        # Component weights
        self.component_weights = self.config.get('component_weights', {
            'greek_sentiment': 0.20,
            'trending_oi_pa': 0.30,
            'iv_skew': 0.20,
            'ema_indicators': 0.15,
            'vwap_indicators': 0.15
        })
        
        # Multi-timeframe parameters
        self.use_multi_timeframe = self.config.get('use_multi_timeframe', True)
        self.timeframe_weights = self.config.get('timeframe_weights', {
            '5m': 0.20,
            '15m': 0.30,
            '1h': 0.30,
            '1d': 0.20
        })
        
        # Time-of-day adjustments
        self.use_time_of_day_adjustments = self.config.get('use_time_of_day_adjustments', True)
        self.time_of_day_weights = self.config.get('time_of_day_weights', {
            'opening': 1.2,  # 9:15-9:45
            'morning': 1.0,  # 9:45-12:00
            'lunch': 0.8,    # 12:00-13:00
            'afternoon': 1.0, # 13:00-14:30
            'closing': 1.2   # 14:30-15:30
        })
        
        logger.info(f"Initialized Market Regime Pipeline with component weights: {self.component_weights}")
    
    def _initialize_components(self):
        """
        Initialize all components.
        """
        # Initialize feature engineering components
        self.greek_sentiment = GreekSentimentAnalysis(self.config.get('greek_sentiment_config', {}))
        self.trending_oi_pa = TrendingOIWithPAAnalysis(self.config.get('trending_oi_pa_config', {}))
        self.iv_skew = IVSkewAnalysis(self.config.get('iv_skew_config', {}))
        self.ema_indicators = EMAIndicators(self.config.get('ema_indicators_config', {}))
        self.vwap_indicators = VWAPIndicators(self.config.get('vwap_indicators_config', {}))
        
        logger.info("Initialized all components")
    
    def process_data(self, data, dte=None, timeframe='5m'):
        """
        Process data through the pipeline.
        
        Args:
            data (pd.DataFrame): Input data
            dte (int, optional): Days to expiry
            timeframe (str, optional): Timeframe of the data
            
        Returns:
            pd.DataFrame: Processed data with market regime
        """
        try:
            logger.info(f"Processing data with {len(data)} rows, timeframe: {timeframe}, DTE: {dte}")
            
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure datetime is in datetime format
            if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except:
                    logger.warning("Failed to convert datetime column to datetime format")
            
            # Apply feature engineering components
            df = self._apply_feature_engineering(df, dte)
            
            # Identify market regime
            df = self._identify_market_regime(df)
            
            # Apply multi-timeframe analysis if enabled
            if self.use_multi_timeframe and 'multi_timeframe_data' in self.config:
                df = self._apply_multi_timeframe_analysis(df, self.config['multi_timeframe_data'])
            
            # Apply time-of-day adjustments if enabled
            if self.use_time_of_day_adjustments and 'datetime' in df.columns:
                df = self._apply_time_of_day_adjustments(df)
            
            # Save results
            self._save_results(df, timeframe, dte)
            
            # Generate visualizations
            self._generate_visualizations(df, timeframe, dte)
            
            logger.info(f"Completed processing data, identified market regimes")
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(traceback.format_exc())
            return data
    
    def _apply_feature_engineering(self, data, dte=None):
        """
        Apply all feature engineering components.
        
        Args:
            data (pd.DataFrame): Input data
            dte (int, optional): Days to expiry
            
        Returns:
            pd.DataFrame: Data with feature engineering applied
        """
        logger.info("Applying feature engineering components")
        
        # Make a copy
        df = data.copy()
        
        try:
            # Apply Greek sentiment analysis
            df = self.greek_sentiment.analyze_greek_sentiment(df, dte)
            logger.info("Applied Greek sentiment analysis")
        except Exception as e:
            logger.error(f"Error applying Greek sentiment analysis: {str(e)}")
        
        try:
            # Apply trending OI with PA analysis
            df = self.trending_oi_pa.analyze_oi_patterns(df)
            logger.info("Applied trending OI with PA analysis")
        except Exception as e:
            logger.error(f"Error applying trending OI with PA analysis: {str(e)}")
        
        try:
            # Apply IV skew analysis
            df = self.iv_skew.analyze_iv_skew(df)
            logger.info("Applied IV skew analysis")
        except Exception as e:
            logger.error(f"Error applying IV skew analysis: {str(e)}")
        
        try:
            # Apply EMA indicators
            df = self.ema_indicators.calculate_ema_indicators(df)
            logger.info("Applied EMA indicators")
        except Exception as e:
            logger.error(f"Error applying EMA indicators: {str(e)}")
        
        try:
            # Apply VWAP indicators
            df = self.vwap_indicators.calculate_vwap_indicators(df)
            logger.info("Applied VWAP indicators")
        except Exception as e:
            logger.error(f"Error applying VWAP indicators: {str(e)}")
        
        return df
    
    def _identify_market_regime(self, data):
        """
        Identify market regime based on all components.
        
        Args:
            data (pd.DataFrame): Data with feature engineering applied
            
        Returns:
            pd.DataFrame: Data with market regime identified
        """
        logger.info("Identifying market regime")
        
        # Make a copy
        df = data.copy()
        
        # Add market regime column
        df['market_regime'] = None
        df['market_regime_confidence'] = 0.0
        df['component_contributions'] = None
        
        # Process each row
        for i, row in df.iterrows():
            # Get component signals
            component_signals = {}
            
            # Greek sentiment signal
            if 'greek_sentiment' in row:
                component_signals['greek_sentiment'] = row['greek_sentiment']
            
            # Trending OI with PA signal
            if 'overall_oi_pattern' in row:
                component_signals['trending_oi_pa'] = row['overall_oi_pattern']
            
            # IV skew signal
            if 'iv_skew_regime' in row:
                component_signals['iv_skew'] = row['iv_skew_regime']
            
            # EMA indicators signal
            if 'ema_regime' in row:
                component_signals['ema_indicators'] = row['ema_regime']
            
            # VWAP indicators signal
            if 'vwap_regime' in row:
                component_signals['vwap_indicators'] = row['vwap_regime']
            
            # Calculate weighted market regime
            regime_result = self._calculate_weighted_regime(component_signals)
            
            # Update dataframe
            df.at[i, 'market_regime'] = regime_result['regime']
            df.at[i, 'market_regime_confidence'] = regime_result['confidence']
            df.at[i, 'component_contributions'] = json.dumps(regime_result['contributions'])
        
        return df
    
    def _calculate_weighted_regime(self, component_signals):
        """
        Calculate weighted market regime from component signals.
        
        Args:
            component_signals (dict): Component signals
            
        Returns:
            dict: Market regime result
        """
        # Initialize regime scores
        regime_scores = {regime: 0.0 for regime in self.market_regimes}
        
        # Initialize component contributions
        contributions = {}
        
        # Calculate total weight of available components
        available_components = set(component_signals.keys()) & set(self.component_weights.keys())
        total_weight = sum(self.component_weights[comp] for comp in available_components)
        
        if total_weight == 0:
            logger.warning("No component weights available, using equal weights")
            total_weight = 1.0
        
        # Process each component
        for component, signal in component_signals.items():
            # Skip if component not in weights or signal is None
            if component not in self.component_weights or signal is None:
                continue
            
            # Get component weight
            weight = self.component_weights[component] / total_weight
            
            # Map signal to regimes
            regime_contribution = self._map_signal_to_regimes(signal, component)
            
            # Add weighted contribution to regime scores
            for regime, score in regime_contribution.items():
                regime_scores[regime] += score * weight
            
            # Store contribution
            contributions[component] = {
                'signal': signal,
                'weight': weight,
                'contribution': regime_contribution
            }
        
        # Find regime with highest score
        if regime_scores:
            top_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime = top_regime[0]
            confidence = top_regime[1]
        else:
            regime = 'Neutral'
            confidence = 0.0
        
        return {
            'regime': regime,
            'confidence': confidence,
            'scores': regime_scores,
            'contributions': contributions
        }
    
    def _map_signal_to_regimes(self, signal, component):
        """
        Map component signal to regime scores.
        
        Args:
            signal (str): Component signal
            component (str): Component name
            
        Returns:
            dict: Regime scores
        """
        # Initialize regime scores
        regime_scores = {regime: 0.0 for regime in self.market_regimes}
        
        # Greek sentiment mapping
        if component == 'greek_sentiment':
            if signal == 'Strong_Bullish':
                regime_scores['Strong_Bullish'] = 1.0
                regime_scores['Bullish'] = 0.7
            elif signal == 'Bullish':
                regime_scores['Bullish'] = 1.0
                regime_scores['Mild_Bullish'] = 0.7
                regime_scores['Strong_Bullish'] = 0.5
            elif signal == 'Mild_Bullish':
                regime_scores['Mild_Bullish'] = 1.0
                regime_scores['Sideways_To_Bullish'] = 0.7
                regime_scores['Bullish'] = 0.5
            elif signal == 'Neutral':
                regime_scores['Neutral'] = 1.0
                regime_scores['Sideways'] = 0.8
                regime_scores['Sideways_To_Bullish'] = 0.4
                regime_scores['Sideways_To_Bearish'] = 0.4
            elif signal == 'Mild_Bearish':
                regime_scores['Mild_Bearish'] = 1.0
                regime_scores['Sideways_To_Bearish'] = 0.7
                regime_scores['Bearish'] = 0.5
            elif signal == 'Bearish':
                regime_scores['Bearish'] = 1.0
                regime_scores['Mild_Bearish'] = 0.7
                regime_scores['Strong_Bearish'] = 0.5
            elif signal == 'Strong_Bearish':
                regime_scores['Strong_Bearish'] = 1.0
                regime_scores['Bearish'] = 0.7
        
        # Trending OI with PA mapping
        elif component == 'trending_oi_pa':
            if signal == 'Strong_Bullish':
                regime_scores['Strong_Bullish'] = 1.0
                regime_scores['Bullish'] = 0.6
            elif signal == 'Mild_Bullish':
                regime_scores['Mild_Bullish'] = 1.0
                regime_scores['Bullish'] = 0.6
                regime_scores['Sideways_To_Bullish'] = 0.4
            elif signal == 'Neutral':
                regime_scores['Neutral'] = 1.0
                regime_scores['Sideways'] = 0.8
            elif signal == 'Mild_Bearish':
                regime_scores['Mild_Bearish'] = 1.0
                regime_scores['Bearish'] = 0.6
                regime_scores['Sideways_To_Bearish'] = 0.4
            elif signal == 'Strong_Bearish':
                regime_scores['Strong_Bearish'] = 1.0
                regime_scores['Bearish'] = 0.6
            elif signal == 'Sideways_To_Bullish':
                regime_scores['Sideways_To_Bullish'] = 1.0
                regime_scores['Mild_Bullish'] = 0.6
                regime_scores['Sideways'] = 0.4
            elif signal == 'Sideways_To_Bearish':
                regime_scores['Sideways_To_Bearish'] = 1.0
                regime_scores['Mild_Bearish'] = 0.6
                regime_scores['Sideways'] = 0.4
            elif signal == 'Sideways':
                regime_scores['Sideways'] = 1.0
                regime_scores['Neutral'] = 0.8
            # Handle reversal patterns
            elif 'Reversal' in signal:
                if 'Bullish' in signal:
                    regime_scores['Reversal_Imminent_Bullish'] = 1.0
                    regime_scores['Failed_Breakout_Bearish'] = 0.6
                elif 'Bearish' in signal:
                    regime_scores['Reversal_Imminent_Bearish'] = 1.0
                    regime_scores['Failed_Breakout_Bullish'] = 0.6
        
        # IV skew mapping
        elif component == 'iv_skew':
            if signal == 'High_Call_Skew':
                regime_scores['Bullish'] = 0.8
                regime_scores['Strong_Bullish'] = 0.6
                regime_scores['Exhaustion_Bullish'] = 0.4
            elif signal == 'Moderate_Call_Skew':
                regime_scores['Mild_Bullish'] = 0.8
                regime_scores['Bullish'] = 0.5
            elif signal == 'Neutral_Skew':
                regime_scores['Neutral'] = 0.9
                regime_scores['Sideways'] = 0.7
            elif signal == 'Moderate_Put_Skew':
                regime_scores['Mild_Bearish'] = 0.8
                regime_scores['Bearish'] = 0.5
            elif signal == 'High_Put_Skew':
                regime_scores['Bearish'] = 0.8
                regime_scores['Strong_Bearish'] = 0.6
                regime_scores['Exhaustion_Bearish'] = 0.4
            elif signal == 'Call_Skew_Increasing':
                regime_scores['Sideways_To_Bullish'] = 0.7
                regime_scores['Mild_Bullish'] = 0.5
            elif signal == 'Put_Skew_Increasing':
                regime_scores['Sideways_To_Bearish'] = 0.7
                regime_scores['Mild_Bearish'] = 0.5
            elif signal == 'Call_Skew_Decreasing':
                regime_scores['Mild_Bearish'] = 0.6
                regime_scores['Reversal_Imminent_Bearish'] = 0.4
            elif signal == 'Put_Skew_Decreasing':
                regime_scores['Mild_Bullish'] = 0.6
                regime_scores['Reversal_Imminent_Bullish'] = 0.4
        
        # EMA indicators mapping
        elif component == 'ema_indicators':
            if signal == 'Strong_Uptrend':
                regime_scores['Strong_Bullish'] = 0.9
                regime_scores['Bullish'] = 0.7
            elif signal == 'Uptrend':
                regime_scores['Bullish'] = 0.9
                regime_scores['Mild_Bullish'] = 0.6
            elif signal == 'Weak_Uptrend':
                regime_scores['Mild_Bullish'] = 0.8
                regime_scores['Sideways_To_Bullish'] = 0.6
            elif signal == 'Sideways':
                regime_scores['Sideways'] = 0.9
                regime_scores['Neutral'] = 0.7
            elif signal == 'Weak_Downtrend':
                regime_scores['Mild_Bearish'] = 0.8
                regime_scores['Sideways_To_Bearish'] = 0.6
            elif signal == 'Downtrend':
                regime_scores['Bearish'] = 0.9
                regime_scores['Mild_Bearish'] = 0.6
            elif signal == 'Strong_Downtrend':
                regime_scores['Strong_Bearish'] = 0.9
                regime_scores['Bearish'] = 0.7
            elif signal == 'Bullish_Crossover':
                regime_scores['Reversal_Imminent_Bullish'] = 0.8
                regime_scores['Mild_Bullish'] = 0.5
            elif signal == 'Bearish_Crossover':
                regime_scores['Reversal_Imminent_Bearish'] = 0.8
                regime_scores['Mild_Bearish'] = 0.5
        
        # VWAP indicators mapping
        elif component == 'vwap_indicators':
            if signal == 'Above_VWAP_Increasing':
                regime_scores['Bullish'] = 0.8
                regime_scores['Mild_Bullish'] = 0.6
            elif signal == 'Above_VWAP_Stable':
                regime_scores['Mild_Bullish'] = 0.7
                regime_scores['Sideways_To_Bullish'] = 0.5
            elif signal == 'At_VWAP':
                regime_scores['Neutral'] = 0.8
                regime_scores['Sideways'] = 0.6
            elif signal == 'Below_VWAP_Stable':
                regime_scores['Mild_Bearish'] = 0.7
                regime_scores['Sideways_To_Bearish'] = 0.5
            elif signal == 'Below_VWAP_Decreasing':
                regime_scores['Bearish'] = 0.8
                regime_scores['Mild_Bearish'] = 0.6
            elif signal == 'VWAP_Breakout_Up':
                regime_scores['Reversal_Imminent_Bullish'] = 0.7
                regime_scores['Mild_Bullish'] = 0.5
            elif signal == 'VWAP_Breakout_Down':
                regime_scores['Reversal_Imminent_Bearish'] = 0.7
                regime_scores['Mild_Bearish'] = 0.5
            elif signal == 'VWAP_Rejection_Up':
                regime_scores['Failed_Breakout_Bullish'] = 0.7
                regime_scores['Mild_Bearish'] = 0.4
            elif signal == 'VWAP_Rejection_Down':
                regime_scores['Failed_Breakout_Bearish'] = 0.7
                regime_scores['Mild_Bullish'] = 0.4
        
        # Default mapping for unknown signals
        else:
            # Try to map based on signal name
            if 'bullish' in signal.lower() or 'uptrend' in signal.lower():
                if 'strong' in signal.lower():
                    regime_scores['Strong_Bullish'] = 0.8
                    regime_scores['Bullish'] = 0.6
                elif 'mild' in signal.lower() or 'weak' in signal.lower():
                    regime_scores['Mild_Bullish'] = 0.8
                    regime_scores['Sideways_To_Bullish'] = 0.5
                else:
                    regime_scores['Bullish'] = 0.8
                    regime_scores['Mild_Bullish'] = 0.5
            elif 'bearish' in signal.lower() or 'downtrend' in signal.lower():
                if 'strong' in signal.lower():
                    regime_scores['Strong_Bearish'] = 0.8
                    regime_scores['Bearish'] = 0.6
                elif 'mild' in signal.lower() or 'weak' in signal.lower():
                    regime_scores['Mild_Bearish'] = 0.8
                    regime_scores['Sideways_To_Bearish'] = 0.5
                else:
                    regime_scores['Bearish'] = 0.8
                    regime_scores['Mild_Bearish'] = 0.5
            elif 'neutral' in signal.lower() or 'sideways' in signal.lower():
                regime_scores['Neutral'] = 0.8
                regime_scores['Sideways'] = 0.6
            elif 'reversal' in signal.lower():
                if 'bullish' in signal.lower():
                    regime_scores['Reversal_Imminent_Bullish'] = 0.8
                    regime_scores['Failed_Breakout_Bearish'] = 0.4
                elif 'bearish' in signal.lower():
                    regime_scores['Reversal_Imminent_Bearish'] = 0.8
                    regime_scores['Failed_Breakout_Bullish'] = 0.4
            else:
                # Default to neutral if can't map
                regime_scores['Neutral'] = 0.5
                regime_scores['Sideways'] = 0.4
        
        return regime_scores
    
    def _apply_multi_timeframe_analysis(self, data, multi_timeframe_data):
        """
        Apply multi-timeframe analysis.
        
        Args:
            data (pd.DataFrame): Data with market regime identified
            multi_timeframe_data (dict): Market regime data for multiple timeframes
            
        Returns:
            pd.DataFrame: Data with multi-timeframe analysis applied
        """
        logger.info("Applying multi-timeframe analysis")
        
        # Make a copy
        df = data.copy()
        
        # Check if multi_timeframe_data is valid
        if not isinstance(multi_timeframe_data, dict) or not multi_timeframe_data:
            logger.warning("Invalid multi-timeframe data, skipping analysis")
            return df
        
        # Add multi-timeframe columns
        df['multi_timeframe_regime'] = None
        df['multi_timeframe_confidence'] = 0.0
        df['timeframe_agreement'] = 0.0
        
        # Process each row
        for i, row in df.iterrows():
            datetime_val = row['datetime']
            primary_regime = row['market_regime']
            
            # Check each timeframe
            timeframe_regimes = {}
            
            for timeframe, timeframe_data in multi_timeframe_data.items():
                # Skip if timeframe is not in weights
                if timeframe not in self.timeframe_weights:
                    continue
                
                # Find closest datetime in this timeframe
                if not pd.api.types.is_datetime64_any_dtype(timeframe_data['datetime']):
                    try:
                        timeframe_data['datetime'] = pd.to_datetime(timeframe_data['datetime'])
                    except:
                        logger.warning(f"Failed to convert datetime column for timeframe {timeframe}")
                        continue
                
                # Find closest datetime
                closest_idx = (timeframe_data['datetime'] - datetime_val).abs().idxmin()
                timeframe_regime = timeframe_data.loc[closest_idx, 'market_regime']
                
                timeframe_regimes[timeframe] = timeframe_regime
            
            # Calculate multi-timeframe regime
            if timeframe_regimes:
                multi_timeframe_result = self._calculate_multi_timeframe_regime(timeframe_regimes, primary_regime)
                
                df.at[i, 'multi_timeframe_regime'] = multi_timeframe_result['regime']
                df.at[i, 'multi_timeframe_confidence'] = multi_timeframe_result['confidence']
                df.at[i, 'timeframe_agreement'] = multi_timeframe_result['agreement']
                
                # Update market regime if multi-timeframe confidence is high
                if multi_timeframe_result['confidence'] > row['market_regime_confidence']:
                    df.at[i, 'market_regime'] = multi_timeframe_result['regime']
                    df.at[i, 'market_regime_confidence'] = multi_timeframe_result['confidence']
        
        return df
    
    def _calculate_multi_timeframe_regime(self, timeframe_regimes, primary_regime):
        """
        Calculate multi-timeframe regime.
        
        Args:
            timeframe_regimes (dict): Regimes for each timeframe
            primary_regime (str): Primary timeframe regime
            
        Returns:
            dict: Multi-timeframe regime result
        """
        # Count regime occurrences
        regime_counts = {}
        
        for timeframe, regime in timeframe_regimes.items():
            if regime not in regime_counts:
                regime_counts[regime] = {
                    'count': 0,
                    'weight': 0.0
                }
            
            regime_counts[regime]['count'] += 1
            regime_counts[regime]['weight'] += self.timeframe_weights.get(timeframe, 0.0)
        
        # Calculate agreement score
        total_weight = sum(self.timeframe_weights.get(timeframe, 0.0) for timeframe in timeframe_regimes.keys())
        
        if primary_regime in regime_counts:
            primary_weight = regime_counts[primary_regime]['weight']
            agreement = primary_weight / total_weight if total_weight > 0 else 0.0
        else:
            agreement = 0.0
        
        # Find regime with highest weight
        if regime_counts:
            top_regime = max(regime_counts.items(), key=lambda x: x[1]['weight'])
            regime = top_regime[0]
            confidence = top_regime[1]['weight'] / total_weight if total_weight > 0 else 0.0
        else:
            regime = primary_regime
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'agreement': agreement,
            'regime_counts': regime_counts
        }
    
    def _apply_time_of_day_adjustments(self, data):
        """
        Apply time-of-day adjustments.
        
        Args:
            data (pd.DataFrame): Data with market regime identified
            
        Returns:
            pd.DataFrame: Data with time-of-day adjustments applied
        """
        logger.info("Applying time-of-day adjustments")
        
        # Make a copy
        df = data.copy()
        
        # Add time-of-day column
        df['time_of_day'] = None
        df['time_adjustment_factor'] = 1.0
        
        # Process each row
        for i, row in df.iterrows():
            datetime_val = row['datetime']
            
            # Get time of day
            time_of_day = self._get_time_of_day(datetime_val)
            
            # Get adjustment factor
            adjustment_factor = self.time_of_day_weights.get(time_of_day, 1.0)
            
            # Update dataframe
            df.at[i, 'time_of_day'] = time_of_day
            df.at[i, 'time_adjustment_factor'] = adjustment_factor
            
            # Adjust confidence
            if 'market_regime_confidence' in df.columns:
                df.at[i, 'market_regime_confidence'] *= adjustment_factor
        
        return df
    
    def _get_time_of_day(self, datetime_val):
        """
        Get time of day category.
        
        Args:
            datetime_val (datetime): Datetime value
            
        Returns:
            str: Time of day category
        """
        hour = datetime_val.hour
        minute = datetime_val.minute
        time_val = hour + minute / 60.0
        
        if 9.25 <= time_val < 9.75:  # 9:15-9:45
            return 'opening'
        elif 9.75 <= time_val < 12.0:  # 9:45-12:00
            return 'morning'
        elif 12.0 <= time_val < 13.0:  # 12:00-13:00
            return 'lunch'
        elif 13.0 <= time_val < 14.5:  # 13:00-14:30
            return 'afternoon'
        elif 14.5 <= time_val < 15.5:  # 14:30-15:30
            return 'closing'
        else:
            return 'after_hours'
    
    def _save_results(self, data, timeframe, dte):
        """
        Save results to file.
        
        Args:
            data (pd.DataFrame): Processed data
            timeframe (str): Timeframe of the data
            dte (int, optional): Days to expiry
        """
        try:
            # Create filename
            dte_str = f"_dte{dte}" if dte is not None else ""
            filename = f"market_regime_{timeframe}{dte_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Save to CSV
            data.to_csv(os.path.join(self.results_dir, filename), index=False)
            
            logger.info(f"Saved results to {filename}")
            
            # Save regime summary
            self._save_regime_summary(data, timeframe, dte)
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _save_regime_summary(self, data, timeframe, dte):
        """
        Save regime summary to file.
        
        Args:
            data (pd.DataFrame): Processed data
            timeframe (str): Timeframe of the data
            dte (int, optional): Days to expiry
        """
        try:
            # Create summary
            regime_counts = data['market_regime'].value_counts()
            regime_confidence = data.groupby('market_regime')['market_regime_confidence'].mean()
            
            summary = pd.DataFrame({
                'count': regime_counts,
                'percentage': regime_counts / len(data) * 100,
                'avg_confidence': regime_confidence
            }).reset_index()
            
            summary = summary.rename(columns={'index': 'market_regime'})
            
            # Create filename
            dte_str = f"_dte{dte}" if dte is not None else ""
            filename = f"regime_summary_{timeframe}{dte_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Save to CSV
            summary.to_csv(os.path.join(self.results_dir, filename), index=False)
            
            logger.info(f"Saved regime summary to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving regime summary: {str(e)}")
    
    def _generate_visualizations(self, data, timeframe, dte):
        """
        Generate visualizations.
        
        Args:
            data (pd.DataFrame): Processed data
            timeframe (str): Timeframe of the data
            dte (int, optional): Days to expiry
        """
        try:
            # Create filename prefix
            dte_str = f"_dte{dte}" if dte is not None else ""
            prefix = f"{timeframe}{dte_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 1. Market regime distribution
            plt.figure(figsize=(12, 8))
            regime_counts = data['market_regime'].value_counts()
            regime_counts.plot(kind='bar')
            plt.title(f'Market Regime Distribution ({timeframe})')
            plt.xlabel('Market Regime')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualization_dir, f"regime_distribution_{prefix}.png"))
            plt.close()
            
            # 2. Market regime over time
            if 'datetime' in data.columns and len(data) > 1:
                plt.figure(figsize=(14, 8))
                
                # Create numeric mapping for regimes
                regimes = data['market_regime'].unique()
                regime_map = {regime: i for i, regime in enumerate(regimes)}
                
                # Convert regimes to numeric values
                numeric_regimes = data['market_regime'].map(regime_map)
                
                # Plot
                plt.scatter(data['datetime'], numeric_regimes, alpha=0.7)
                plt.yticks(range(len(regimes)), regimes)
                plt.title(f'Market Regime Over Time ({timeframe})')
                plt.xlabel('Datetime')
                plt.ylabel('Market Regime')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualization_dir, f"regime_over_time_{prefix}.png"))
                plt.close()
            
            # 3. Component contribution heatmap
            if 'component_contributions' in data.columns:
                # Extract component contributions
                components = set()
                for _, row in data.iterrows():
                    if pd.notna(row['component_contributions']):
                        contributions = json.loads(row['component_contributions'])
                        components.update(contributions.keys())
                
                components = sorted(components)
                
                if components:
                    # Create contribution matrix
                    contribution_matrix = np.zeros((len(components), len(self.market_regimes)))
                    
                    for comp_idx, component in enumerate(components):
                        for regime_idx, regime in enumerate(self.market_regimes):
                            # Calculate average contribution of this component to this regime
                            total_contribution = 0.0
                            count = 0
                            
                            for _, row in data.iterrows():
                                if pd.notna(row['component_contributions']):
                                    contributions = json.loads(row['component_contributions'])
                                    
                                    if component in contributions:
                                        comp_contrib = contributions[component]
                                        
                                        if 'contribution' in comp_contrib:
                                            regime_contrib = comp_contrib['contribution'].get(regime, 0.0)
                                            total_contribution += regime_contrib
                                            count += 1
                            
                            if count > 0:
                                contribution_matrix[comp_idx, regime_idx] = total_contribution / count
                    
                    # Plot heatmap
                    plt.figure(figsize=(16, 10))
                    sns.heatmap(
                        contribution_matrix,
                        xticklabels=self.market_regimes,
                        yticklabels=components,
                        cmap='viridis',
                        annot=False
                    )
                    plt.title(f'Component Contribution to Market Regimes ({timeframe})')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Component')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.visualization_dir, f"component_contribution_{prefix}.png"))
                    plt.close()
            
            # 4. Confidence distribution
            if 'market_regime_confidence' in data.columns:
                plt.figure(figsize=(10, 6))
                data['market_regime_confidence'].hist(bins=20)
                plt.title(f'Market Regime Confidence Distribution ({timeframe})')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualization_dir, f"confidence_distribution_{prefix}.png"))
                plt.close()
            
            # 5. Time-of-day analysis
            if 'time_of_day' in data.columns:
                plt.figure(figsize=(12, 8))
                time_regime_counts = data.groupby(['time_of_day', 'market_regime']).size().unstack(fill_value=0)
                time_regime_counts.plot(kind='bar', stacked=True)
                plt.title(f'Market Regime by Time of Day ({timeframe})')
                plt.xlabel('Time of Day')
                plt.ylabel('Count')
                plt.legend(title='Market Regime')
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualization_dir, f"regime_by_time_{prefix}.png"))
                plt.close()
            
            logger.info(f"Generated visualizations with prefix {prefix}")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def prepare_for_consolidator(self, data, dte=None):
        """
        Prepare data for consolidator.
        
        Args:
            data (pd.DataFrame): Processed data with market regime
            dte (int, optional): Days to expiry
            
        Returns:
            pd.DataFrame: Data formatted for consolidator
        """
        try:
            logger.info("Preparing data for consolidator")
            
            # Make a copy
            df = data.copy()
            
            # Ensure datetime is in datetime format
            if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except:
                    logger.warning("Failed to convert datetime column to datetime format")
            
            # Extract date and time
            if 'datetime' in df.columns:
                df['Date'] = df['datetime'].dt.date
                df['Time'] = df['datetime'].dt.time
            
            # Add DTE column if provided
            if dte is not None:
                df['DTE'] = dte
            
            # Add day of week
            if 'datetime' in df.columns:
                df['Day'] = df['datetime'].dt.day_name()
            
            # Add Zone column (time of day)
            if 'time_of_day' in df.columns:
                df['Zone'] = df['time_of_day']
            elif 'datetime' in df.columns:
                df['Zone'] = df['datetime'].apply(lambda x: self._get_time_of_day(x))
            
            # Ensure market_regime column exists
            if 'market_regime' not in df.columns:
                logger.warning("No market_regime column found, using Neutral as default")
                df['market_regime'] = 'Neutral'
            
            # Add confidence column if not exists
            if 'market_regime_confidence' not in df.columns:
                df['market_regime_confidence'] = 0.5
            
            # Select and reorder columns for consolidator
            consolidator_columns = [
                'Date', 'Time', 'Zone', 'DTE', 'Day', 'market_regime', 
                'market_regime_confidence'
            ]
            
            # Add any strategy performance columns if they exist
            strategy_columns = [col for col in df.columns if 'Strategy' in col and 'performance' in col]
            consolidator_columns.extend(strategy_columns)
            
            # Select only columns that exist in the dataframe
            existing_columns = [col for col in consolidator_columns if col in df.columns]
            
            # Create consolidator dataframe
            consolidator_df = df[existing_columns].copy()
            
            logger.info(f"Prepared data for consolidator with columns: {existing_columns}")
            
            return consolidator_df
        
        except Exception as e:
            logger.error(f"Error preparing data for consolidator: {str(e)}")
            return data
    
    def run_pipeline(self, data_file, output_file=None, dte=None, timeframe='5m'):
        """
        Run the complete pipeline on a data file.
        
        Args:
            data_file (str): Path to input data file
            output_file (str, optional): Path to output file
            dte (int, optional): Days to expiry
            timeframe (str, optional): Timeframe of the data
            
        Returns:
            pd.DataFrame: Processed data with market regime
        """
        try:
            logger.info(f"Running pipeline on {data_file}")
            
            # Load data
            if data_file.endswith('.csv'):
                data = pd.read_csv(data_file)
            elif data_file.endswith('.xlsx') or data_file.endswith('.xls'):
                data = pd.read_excel(data_file)
            else:
                logger.error(f"Unsupported file format: {data_file}")
                return None
            
            # Process data
            processed_data = self.process_data(data, dte, timeframe)
            
            # Prepare for consolidator
            consolidator_data = self.prepare_for_consolidator(processed_data, dte)
            
            # Save output if specified
            if output_file:
                consolidator_data.to_csv(output_file, index=False)
                logger.info(f"Saved consolidator data to {output_file}")
            
            return consolidator_data
        
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def run_multi_timeframe_pipeline(self, data_files, output_dir=None, dte=None):
        """
        Run the pipeline on multiple timeframes.
        
        Args:
            data_files (dict): Dictionary mapping timeframes to data files
            output_dir (str, optional): Output directory
            dte (int, optional): Days to expiry
            
        Returns:
            dict: Dictionary mapping timeframes to processed data
        """
        try:
            logger.info(f"Running multi-timeframe pipeline on {len(data_files)} timeframes")
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Process each timeframe
            results = {}
            multi_timeframe_data = {}
            
            for timeframe, data_file in data_files.items():
                logger.info(f"Processing timeframe {timeframe}")
                
                # Create output file path if specified
                output_file = None
                if output_dir:
                    dte_str = f"_dte{dte}" if dte is not None else ""
                    output_file = os.path.join(output_dir, f"consolidator_{timeframe}{dte_str}.csv")
                
                # Run pipeline
                processed_data = self.run_pipeline(data_file, output_file, dte, timeframe)
                
                if processed_data is not None:
                    results[timeframe] = processed_data
                    
                    # Store for multi-timeframe analysis
                    multi_timeframe_data[timeframe] = processed_data
            
            # Update config with multi-timeframe data
            self.config['multi_timeframe_data'] = multi_timeframe_data
            
            # Re-run with multi-timeframe analysis
            for timeframe, data_file in data_files.items():
                if timeframe in results:
                    logger.info(f"Re-processing timeframe {timeframe} with multi-timeframe analysis")
                    
                    # Load data
                    if data_file.endswith('.csv'):
                        data = pd.read_csv(data_file)
                    elif data_file.endswith('.xlsx') or data_file.endswith('.xls'):
                        data = pd.read_excel(data_file)
                    else:
                        continue
                    
                    # Process data with multi-timeframe analysis
                    processed_data = self.process_data(data, dte, timeframe)
                    
                    # Prepare for consolidator
                    consolidator_data = self.prepare_for_consolidator(processed_data, dte)
                    
                    # Save output if specified
                    if output_dir:
                        dte_str = f"_dte{dte}" if dte is not None else ""
                        output_file = os.path.join(output_dir, f"consolidator_{timeframe}{dte_str}_multi.csv")
                        consolidator_data.to_csv(output_file, index=False)
                        logger.info(f"Saved multi-timeframe consolidator data to {output_file}")
                    
                    # Update results
                    results[timeframe] = consolidator_data
            
            return results
        
        except Exception as e:
            logger.error(f"Error running multi-timeframe pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
"""
