"""
Memory-Optimized Minute Regime Classifier

This module implements minute-by-minute market regime classification with multiple indicators
at different timeframes and dynamic weightage based on historical performance.

This version includes optimizations for memory efficiency to handle large datasets.
"""

import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import sys
import time
import gc
import psutil
import warnings

# Add project root to path
sys.path.append('/home/ubuntu/market_regime_testing')

# Import indicator modules
from utils.feature_engineering.trending_oi_pa import TrendingOIWithPAAnalysis
from utils.feature_engineering.ema_indicators import EMAIndicators
from utils.feature_engineering.vwap_indicators import VWAPIndicators
from utils.feature_engineering.greek_sentiment import GreekSentimentAnalysis
from utils.dynamic_weight_adjustment.dynamic_weight_adjustment import DynamicWeightAdjustment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("minute_regime_classifier.log")
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MemoryMonitor:
    """
    A utility class to monitor memory usage during processing.
    """
    
    @staticmethod
    def get_memory_usage():
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        return memory_usage_mb
    
    @staticmethod
    def log_memory_usage(label="Current"):
        """
        Log current memory usage.
        
        Args:
            label (str): Label for the log message
        """
        memory_usage_mb = MemoryMonitor.get_memory_usage()
        logger.info(f"{label} memory usage: {memory_usage_mb:.2f} MB")
    
    @staticmethod
    def check_memory_threshold(threshold_mb=1000):
        """
        Check if memory usage exceeds threshold.
        
        Args:
            threshold_mb (int): Memory threshold in MB
            
        Returns:
            bool: True if memory usage exceeds threshold, False otherwise
        """
        memory_usage_mb = MemoryMonitor.get_memory_usage()
        return memory_usage_mb > threshold_mb

class MinuteRegimeClassifier:
    """
    A memory-optimized class for classifying market regimes at minute-level granularity
    with efficient processing for integration with strategy consolidator.
    """
    
    def __init__(self, config=None):
        """
        Initialize the MinuteRegimeClassifier with configuration parameters.
        
        Args:
            config (dict, optional): Configuration parameters for classification.
        """
        self.config = config or self._get_default_config()
        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        
        # Initialize indicator modules
        self.trending_oi_pa = TrendingOIWithPAAnalysis(self.config.get('trending_oi_pa_config'))
        self.ema_indicators = EMAIndicators(self.config.get('ema_indicators_config'))
        self.vwap_indicators = VWAPIndicators(self.config.get('vwap_indicators_config'))
        self.greek_sentiment = GreekSentimentAnalysis(self.config.get('greek_sentiment_config'))
        
        # Initialize dynamic weight adjustment
        self.weight_adjuster = DynamicWeightAdjustment(self.config.get('weight_adjustment_config'))
        
        # Initialize regime mapping for strategy consolidator
        self.regime_strategy_map = self._initialize_regime_strategy_map()
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        self.memory_threshold_mb = self.config.get('memory_threshold_mb', 1000)
        
        logger.info("Initialized Memory-Optimized Minute Regime Classifier")
        logger.info(f"Output directory: {self.output_dir}")
        self.memory_monitor.log_memory_usage("Initial")
        
    def _get_default_config(self):
        """
        Get default configuration parameters.
        
        Returns:
            dict: Default configuration parameters.
        """
        return {
            'output_dir': '/home/ubuntu/market_regime_testing/output/minute_regime_analysis',
            'ema_periods': [20, 50, 200],
            'volatility_window': 20,
            'momentum_window': 14,
            'skew_window': 10,
            'confidence_threshold': 0.7,
            'chunk_size': 1000,  # Reduced chunk size for better memory efficiency
            'memory_threshold_mb': 1000,  # Memory threshold in MB
            'time_periods': {
                'opening': ('09:15', '10:00'),
                'morning': ('10:00', '12:00'),
                'midday': ('12:00', '14:00'),
                'closing': ('14:00', '15:30')
            },
            'trending_oi_pa_config': {
                'default_weight': 0.30,
                'strikes_above_atm': 7,
                'strikes_below_atm': 7
            },
            'ema_indicators_config': {
                'default_weight': 0.20,
                'ema_periods_15m': [20, 50, 200],
                'ema_periods_10m': [20, 50, 200],
                'ema_periods_5m': [20, 50, 200],
                'ema_periods_3m': [20, 50, 200]
            },
            'vwap_indicators_config': {
                'default_weight': 0.20,
                'vwap_band_multipliers': [1.0, 2.0, 3.0]
            },
            'greek_sentiment_config': {
                'default_weight': 0.25,
                'delta_weight': 0.4,
                'gamma_weight': 0.2,
                'theta_weight': 0.2,
                'vega_weight': 0.2
            },
            'weight_adjustment_config': {
                'trending_oi_pa_weight': 0.30,
                'ema_indicators_weight': 0.20,
                'vwap_indicators_weight': 0.20,
                'greek_sentiment_weight': 0.25,
                'other_indicators_weight': 0.05,
                'learning_rate': 0.1,
                'time_of_day_adjustment': True,
                'volatility_adjustment': True
            },
            'date_filter': {
                'start_date': '2025-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d')
            }
        }
    
    def _initialize_regime_strategy_map(self):
        """
        Initialize mapping between market regimes and optimal strategies.
        This will be used by the strategy consolidator.
        
        Returns:
            dict: Mapping between regimes and strategies.
        """
        return {
            # Bullish regimes
            'Bullish_Low_Vol_Balanced': ['long_call', 'bull_call_spread'],
            'Bullish_Low_Vol_Call_Skew': ['bull_call_spread', 'call_ratio_spread'],
            'Bullish_Low_Vol_Put_Skew': ['long_call', 'bull_put_spread'],
            'Bullish_Medium_Vol_Balanced': ['bull_call_spread', 'long_call'],
            'Bullish_Medium_Vol_Call_Skew': ['bull_call_spread', 'call_calendar_spread'],
            'Bullish_Medium_Vol_Put_Skew': ['long_call', 'bull_put_spread'],
            'Bullish_High_Vol_Balanced': ['bull_call_spread', 'long_call_butterfly'],
            'Bullish_High_Vol_Call_Skew': ['bull_call_spread', 'call_ratio_spread'],
            'Bullish_High_Vol_Put_Skew': ['long_call', 'bull_put_spread'],
            
            # Bearish regimes
            'Bearish_Low_Vol_Balanced': ['long_put', 'bear_put_spread'],
            'Bearish_Low_Vol_Call_Skew': ['long_put', 'bear_call_spread'],
            'Bearish_Low_Vol_Put_Skew': ['bear_put_spread', 'put_ratio_spread'],
            'Bearish_Medium_Vol_Balanced': ['bear_put_spread', 'long_put'],
            'Bearish_Medium_Vol_Call_Skew': ['long_put', 'bear_call_spread'],
            'Bearish_Medium_Vol_Put_Skew': ['bear_put_spread', 'put_calendar_spread'],
            'Bearish_High_Vol_Balanced': ['bear_put_spread', 'long_put_butterfly'],
            'Bearish_High_Vol_Call_Skew': ['long_put', 'bear_call_spread'],
            'Bearish_High_Vol_Put_Skew': ['bear_put_spread', 'put_ratio_spread'],
            
            # Neutral regimes
            'Neutral_Low_Vol_Balanced': ['iron_condor', 'short_straddle'],
            'Neutral_Low_Vol_Call_Skew': ['iron_condor', 'call_credit_spread'],
            'Neutral_Low_Vol_Put_Skew': ['iron_condor', 'put_credit_spread'],
            'Neutral_Medium_Vol_Balanced': ['iron_condor', 'calendar_spread'],
            'Neutral_Medium_Vol_Call_Skew': ['iron_condor', 'call_calendar_spread'],
            'Neutral_Medium_Vol_Put_Skew': ['iron_condor', 'put_calendar_spread'],
            'Neutral_High_Vol_Balanced': ['iron_butterfly', 'condor'],
            'Neutral_High_Vol_Call_Skew': ['iron_condor', 'call_calendar_spread'],
            'Neutral_High_Vol_Put_Skew': ['iron_condor', 'put_calendar_spread']
        }
    
    def _filter_by_date_range(self, df):
        """
        Filter dataframe by date range.
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime column
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        try:
            # Get date filter from config
            date_filter = self.config.get('date_filter', {})
            start_date = date_filter.get('start_date')
            end_date = date_filter.get('end_date')
            
            if not start_date or not end_date:
                return df
            
            # Convert to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                # Create datetime column from date and time columns
                if 'date' in df.columns and 'time' in df.columns:
                    # Convert date and time to string first
                    df['date'] = df['date'].astype(str)
                    df['time'] = df['time'].astype(str)
                    
                    # Pad time with leading zeros if needed
                    df['time'] = df['time'].apply(lambda x: x.zfill(4))
                    
                    # Format time with colon
                    df['time'] = df['time'].apply(lambda x: f"{x[:2]}:{x[2:]}" if len(x) == 4 else x)
                    
                    # Combine date and time
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            
            # Filter by date range
            mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
            filtered_df = df[mask].copy()
            
            logger.info(f"Filtered data from {start_date} to {end_date}: {len(filtered_df)} rows")
            return filtered_df
        except Exception as e:
            logger.error(f"Error filtering by date range: {str(e)}")
            return df
    
    def process_file(self, file_path):
        """
        Process a single data file for minute-by-minute regime classification.
        Uses chunking to minimize memory usage.
        
        Args:
            file_path (str): Path to the data file.
            
        Returns:
            pd.DataFrame: Classified regimes at minute level.
        """
        logger.info(f"Processing {file_path} for minute-by-minute regime classification")
        self.memory_monitor.log_memory_usage("Before processing file")
        
        # Read file in chunks to save memory
        chunk_size = self.config['chunk_size']
        
        try:
            # Get total number of rows for progress tracking
            with open(file_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header row
            
            total_chunks = (total_rows // chunk_size) + 1
            
            logger.info(f"Processing {total_rows} rows in {total_chunks} chunks")
            
            # Create output directory for intermediate results
            intermediate_dir = os.path.join(self.output_dir, 'intermediate_results')
            os.makedirs(intermediate_dir, exist_ok=True)
            
            # Process each chunk with progress tracking
            for i, chunk in enumerate(tqdm(pd.read_csv(file_path, chunksize=chunk_size), total=total_chunks, desc="Processing chunks")):
                try:
                    # Check memory usage before processing chunk
                    if self.memory_monitor.check_memory_threshold(self.memory_threshold_mb):
                        logger.warning(f"Memory usage exceeds threshold ({self.memory_threshold_mb} MB). Forcing garbage collection.")
                        gc.collect()
                    
                    # Filter by date range
                    filtered_chunk = self._filter_by_date_range(chunk)
                    
                    # Skip empty chunks
                    if filtered_chunk.empty:
                        logger.info(f"Chunk {i+1}/{total_chunks} is empty after date filtering. Skipping.")
                        continue
                    
                    # Process chunk
                    processed_chunk = self._process_chunk(filtered_chunk)
                    
                    # Skip empty processed chunks
                    if processed_chunk.empty:
                        logger.info(f"Processed chunk {i+1}/{total_chunks} is empty. Skipping.")
                        continue
                    
                    # Save intermediate result
                    intermediate_file = os.path.join(intermediate_dir, f"chunk_{i+1}.csv")
                    processed_chunk.to_csv(intermediate_file, index=False)
                    
                    # Log progress
                    logger.info(f"Processed chunk {i+1}/{total_chunks} ({len(processed_chunk)} rows)")
                    self.memory_monitor.log_memory_usage(f"After chunk {i+1}")
                    
                    # Free memory
                    del chunk
                    del filtered_chunk
                    del processed_chunk
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    # Continue with next chunk
                    continue
            
            # Combine intermediate results
            logger.info("Combining intermediate results")
            self.memory_monitor.log_memory_usage("Before combining results")
            
            # Get list of intermediate files
            intermediate_files = [os.path.join(intermediate_dir, f) for f in os.listdir(intermediate_dir) if f.endswith('.csv')]
            
            if not intermediate_files:
                logger.error("No intermediate files found")
                return pd.DataFrame()
            
            # Combine results in batches to save memory
            batch_size = 5
            combined_results = []
            
            for i in range(0, len(intermediate_files), batch_size):
                batch_files = intermediate_files[i:i+batch_size]
                batch_dfs = []
                
                for file in batch_files:
                    try:
                        df = pd.read_csv(file)
                        batch_dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file}: {str(e)}")
                
                if batch_dfs:
                    batch_combined = pd.concat(batch_dfs, ignore_index=True)
                    
                    # Save batch result
                    batch_file = os.path.join(intermediate_dir, f"batch_{i//batch_size+1}.csv")
                    batch_combined.to_csv(batch_file, index=False)
                    combined_results.append(batch_file)
                    
                    # Free memory
                    del batch_dfs
                    del batch_combined
                    gc.collect()
            
            # Combine batch results
            final_dfs = []
            
            for file in combined_results:
                try:
                    df = pd.read_csv(file)
                    final_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
            
            if not final_dfs:
                logger.error("No final results to combine")
                return pd.DataFrame()
            
            result = pd.concat(final_dfs, ignore_index=True)
            
            # Free memory
            del final_dfs
            gc.collect()
            
            # Save final result
            output_file = os.path.join(self.output_dir, f"minute_regime_classification_{os.path.basename(file_path)}")
            result.to_csv(output_file, index=False)
            logger.info(f"Saved final result to {output_file}")
            self.memory_monitor.log_memory_usage("After saving final result")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _process_chunk(self, chunk):
        """
        Process a chunk of data for minute-by-minute regime classification.
        
        Args:
            chunk (pd.DataFrame): Chunk of data to process.
            
        Returns:
            pd.DataFrame: Processed chunk with regime classifications.
        """
        try:
            # Make a copy to avoid modifying the original
            df = chunk.copy()
            
            # Ensure numeric columns are numeric
            numeric_cols = ['strike', 'underlying_price', 'PE_oi', 'CE_oi', 
                           'put_delta', 'put_gamma', 'put_theta', 'put_vega',
                           'call_delta', 'call_gamma', 'call_theta', 'call_vega']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure datetime format
            if 'datetime' not in df.columns:
                # Create datetime column from date and time columns
                if 'date' in df.columns and 'time' in df.columns:
                    # Convert date and time to string first
                    df['date'] = df['date'].astype(str)
                    df['time'] = df['time'].astype(str)
                    
                    # Pad time with leading zeros if needed
                    df['time'] = df['time'].apply(lambda x: x.zfill(4))
                    
                    # Format time with colon
                    df['time'] = df['time'].apply(lambda x: f"{x[:2]}:{x[2:]}" if len(x) == 4 else x)
                    
                    # Combine date and time
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                else:
                    # Try to find alternative column names
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    time_cols = [col for col in df.columns if 'time' in col.lower()]
                    
                    if date_cols and time_cols:
                        df['date'] = df[date_cols[0]].astype(str)
                        df['time'] = df[time_cols[0]].astype(str)
                        
                        # Pad time with leading zeros if needed
                        df['time'] = df['time'].apply(lambda x: x.zfill(4))
                        
                        # Format time with colon
                        df['time'] = df['time'].apply(lambda x: f"{x[:2]}:{x[2:]}" if len(x) == 4 else x)
                        
                        # Combine date and time
                        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                    else:
                        raise ValueError("Could not identify datetime columns in the data")
            
            # Extract minute-level timestamp
            df['minute'] = df['datetime'].dt.floor('1min')
            
            # Create option_type column if not exists
            if 'option_type' not in df.columns:
                # Try to infer from symbol names
                if 'CE_symbol' in df.columns and 'PE_symbol' in df.columns:
                    # Create option_type column based on which symbol is not NaN
                    df['option_type'] = np.where(df['CE_symbol'].notna(), 'call', 'put')
                    logger.info("Created option_type column based on CE_symbol and PE_symbol")
            
            # Process indicators sequentially to save memory
            # Apply trending OI with PA analysis (5-minute timeframe)
            df = self._apply_trending_oi_pa_analysis(df)
            gc.collect()  # Force garbage collection
            
            # Apply EMA indicators at multiple timeframes
            df = self._apply_ema_indicators(df)
            gc.collect()  # Force garbage collection
            
            # Apply VWAP indicators at multiple timeframes
            df = self._apply_vwap_indicators(df)
            gc.collect()  # Force garbage collection
            
            # Apply Greek sentiment analysis
            df = self._apply_greek_sentiment_analysis(df)
            gc.collect()  # Force garbage collection
            
            # Combine indicators with dynamic weightage
            df = self._combine_indicators_with_dynamic_weights(df)
            gc.collect()  # Force garbage collection
            
            # Classify regimes
            df = self._classify_regimes(df)
            gc.collect()  # Force garbage collection
            
            # Add confidence scores
            df = self._add_confidence_scores(df)
            gc.collect()  # Force garbage collection
            
            # Add strategy recommendations for consolidator
            df = self._add_strategy_recommendations(df)
            gc.collect()  # Force garbage collection
            
            # Select only necessary columns to save memory
            essential_columns = [
                'datetime', 'minute', 'underlying_price', 'strike',
                'trending_oi_pa_component', 'trending_oi_pa_class', 'trending_oi_pa_confidence',
                'ema_regime_component', 'ema_regime_class', 'ema_confidence',
                'vwap_regime_component', 'vwap_regime_class', 'vwap_confidence',
                'greek_regime_component', 'greek_regime_class', 'greek_confidence',
                'combined_regime_component', 'regime', 'confidence_score',
                'trending_oi_pa_weight', 'ema_indicators_weight', 'vwap_indicators_weight', 'greek_sentiment_weight',
                'recommended_strategies'
            ]
            
            # Keep only columns that exist in the dataframe
            columns_to_keep = [col for col in essential_columns if col in df.columns]
            df = df[columns_to_keep].copy()
            
            return df
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            # Return empty DataFrame with same columns as input
            return pd.DataFrame(columns=chunk.columns)
    
    def _apply_trending_oi_pa_analysis(self, data):
        """
        Apply trending OI with PA analysis to the data.
        
        Args:
            data (pd.DataFrame): Data to analyze.
            
        Returns:
            pd.DataFrame: Data with trending OI with PA analysis.
        """
        logger.info("Applying trending OI with PA analysis")
        
        try:
            # Apply trending OI with PA analysis
            result = self.trending_oi_pa.analyze_oi_patterns(data)
            
            # Get regime component
            if 'oi_pattern' in result.columns:
                # Map OI patterns to regime components
                pattern_mapping = {
                    'Long_Build_Up': 0.8,  # Strong bullish
                    'Short_Covering': 0.5,  # Bullish
                    'Short_Build_Up': -0.8,  # Strong bearish
                    'Long_Unwinding': -0.5,  # Bearish
                    'Neutral': 0.0  # Neutral
                }
                
                # Apply mapping
                result['trending_oi_pa_component'] = result['oi_pattern'].map(pattern_mapping).fillna(0)
                
                # Classify trending OI regime component
                result['trending_oi_pa_class'] = 'neutral'
                result.loc[result['trending_oi_pa_component'] > 0.7, 'trending_oi_pa_class'] = 'strong_bullish'
                result.loc[(result['trending_oi_pa_component'] > 0.3) & (result['trending_oi_pa_component'] <= 0.7), 'trending_oi_pa_class'] = 'bullish'
                result.loc[(result['trending_oi_pa_component'] > -0.3) & (result['trending_oi_pa_component'] <= 0.3), 'trending_oi_pa_class'] = 'neutral'
                result.loc[(result['trending_oi_pa_component'] > -0.7) & (result['trending_oi_pa_component'] <= -0.3), 'trending_oi_pa_class'] = 'bearish'
                result.loc[result['trending_oi_pa_component'] <= -0.7, 'trending_oi_pa_class'] = 'strong_bearish'
                
                # Calculate confidence score
                result['trending_oi_pa_confidence'] = result['trending_oi_pa_component'].abs()
            else:
                # If no OI patterns, use a default neutral component
                result['trending_oi_pa_component'] = 0.0
                result['trending_oi_pa_class'] = 'neutral'
                result['trending_oi_pa_confidence'] = 0.5
            
            logger.info("Trending OI with PA analysis applied successfully")
            return result
        except Exception as e:
            logger.error(f"Error applying trending OI with PA analysis: {str(e)}")
            
            # Return original data with default values
            data['trending_oi_pa_component'] = 0.0
            data['trending_oi_pa_class'] = 'neutral'
            data['trending_oi_pa_confidence'] = 0.5
            return data
    
    def _apply_ema_indicators(self, data):
        """
        Apply EMA indicators at multiple timeframes to the data.
        
        Args:
            data (pd.DataFrame): Data to analyze.
            
        Returns:
            pd.DataFrame: Data with EMA indicators.
        """
        logger.info("Applying EMA indicators at multiple timeframes")
        
        try:
            # Ensure we have close price column
            if 'close' not in data.columns:
                # Try to find alternative column names
                close_cols = ['Close', 'CLOSE', 'last_price', 'price', 'underlying_price']
                
                for col in close_cols:
                    if col in data.columns:
                        data['close'] = data[col]
                        logger.info(f"Using {col} as close price")
                        break
                else:
                    # If no close price column found, use underlying_price
                    if 'underlying_price' in data.columns:
                        data['close'] = data['underlying_price']
                        logger.info("Using underlying_price as close price")
                    else:
                        logger.warning("No close price column found")
                        # Add default ema columns
                        data['ema_regime_component'] = 0.0
                        data['ema_regime_class'] = 'neutral'
                        data['ema_confidence'] = 0.5
                        return data
            
            # Apply EMA indicators at multiple timeframes
            timeframes = ['15min', '10min', '5min', '3min']
            result = self.ema_indicators.get_ema_regime_component(data, timeframes)
            
            logger.info("EMA indicators applied successfully")
            return result
        except Exception as e:
            logger.error(f"Error applying EMA indicators: {str(e)}")
            
            # Return original data with default values
            data['ema_regime_component'] = 0.0
            data['ema_regime_class'] = 'neutral'
            data['ema_confidence'] = 0.5
            return data
    
    def _apply_vwap_indicators(self, data):
        """
        Apply VWAP indicators at multiple timeframes to the data.
        
        Args:
            data (pd.DataFrame): Data to analyze.
            
        Returns:
            pd.DataFrame: Data with VWAP indicators.
        """
        logger.info("Applying VWAP indicators at multiple timeframes")
        
        try:
            # Ensure we have volume column
            if 'volume' not in data.columns:
                # Try to find alternative column names
                volume_cols = ['Volume', 'VOLUME', 'vol', 'PE_volume', 'CE_volume']
                
                for col in volume_cols:
                    if col in data.columns:
                        data['volume'] = data[col]
                        logger.info(f"Using {col} as volume")
                        break
                else:
                    # If no volume column found, create a dummy one
                    logger.warning("No volume column found, creating dummy volume")
                    data['volume'] = 1
            
            # Apply VWAP indicators at multiple timeframes
            timeframes = ['15min', '10min', '5min', '3min']
            result = self.vwap_indicators.get_vwap_regime_component(data, timeframes)
            
            logger.info("VWAP indicators applied successfully")
            return result
        except Exception as e:
            logger.error(f"Error applying VWAP indicators: {str(e)}")
            
            # Return original data with default values
            data['vwap_regime_component'] = 0.0
            data['vwap_regime_class'] = 'neutral'
            data['vwap_confidence'] = 0.5
            return data
    
    def _apply_greek_sentiment_analysis(self, data):
        """
        Apply Greek sentiment analysis to the data.
        
        Args:
            data (pd.DataFrame): Data to analyze.
            
        Returns:
            pd.DataFrame: Data with Greek sentiment analysis.
        """
        logger.info("Applying Greek sentiment analysis")
        
        try:
            # Apply Greek sentiment analysis
            result = self.greek_sentiment.get_greek_regime_component(data)
            
            logger.info("Greek sentiment analysis applied successfully")
            return result
        except Exception as e:
            logger.error(f"Error applying Greek sentiment analysis: {str(e)}")
            
            # Return original data with default values
            data['greek_regime_component'] = 0.0
            data['greek_regime_class'] = 'neutral'
            data['greek_confidence'] = 0.5
            return data
    
    def _combine_indicators_with_dynamic_weights(self, data):
        """
        Combine indicators with dynamic weightage.
        
        Args:
            data (pd.DataFrame): Data with indicator components.
            
        Returns:
            pd.DataFrame: Data with combined regime component.
        """
        logger.info("Combining indicators with dynamic weightage")
        
        try:
            # Get current timestamp for time-of-day adjustment
            timestamp = data['datetime'].iloc[0] if not data.empty else None
            
            # Determine volatility level based on data
            if 'volatility' in data.columns:
                volatility = data['volatility'].mean()
                
                # Classify volatility level
                if volatility < 0.01:  # 1% volatility
                    volatility_level = 'low'
                elif volatility < 0.02:  # 2% volatility
                    volatility_level = 'medium'
                else:
                    volatility_level = 'high'
            else:
                volatility_level = None
            
            # Get current weights from dynamic weight adjuster
            weights = self.weight_adjuster.get_current_weights(timestamp, volatility_level)
            
            # Ensure all component columns exist with default values
            component_cols = {
                'trending_oi_pa_component': 0.0,
                'ema_regime_component': 0.0,
                'vwap_regime_component': 0.0,
                'greek_regime_component': 0.0
            }
            
            for col, default_val in component_cols.items():
                if col not in data.columns:
                    data[col] = default_val
            
            # Combine indicator components with weights
            data['combined_regime_component'] = (
                weights.get('trending_oi_pa', 0.30) * data['trending_oi_pa_component'] +
                weights.get('ema_indicators', 0.20) * data['ema_regime_component'] +
                weights.get('vwap_indicators', 0.20) * data['vwap_regime_component'] +
                weights.get('greek_sentiment', 0.25) * data['greek_regime_component'] +
                weights.get('other_indicators', 0.05) * 0  # Placeholder for other indicators
            )
            
            # Normalize to range [-1, 1]
            data['combined_regime_component'] = data['combined_regime_component'].clip(-1, 1)
            
            # Store weights used for reference
            data['trending_oi_pa_weight'] = weights.get('trending_oi_pa', 0.30)
            data['ema_indicators_weight'] = weights.get('ema_indicators', 0.20)
            data['vwap_indicators_weight'] = weights.get('vwap_indicators', 0.20)
            data['greek_sentiment_weight'] = weights.get('greek_sentiment', 0.25)
            data['other_indicators_weight'] = weights.get('other_indicators', 0.05)
            
            logger.info(f"Indicators combined successfully with weights: {weights}")
            return data
        except Exception as e:
            logger.error(f"Error combining indicators: {str(e)}")
            
            # Return original data with default combined component
            data['combined_regime_component'] = 0.0
            data['trending_oi_pa_weight'] = 0.30
            data['ema_indicators_weight'] = 0.20
            data['vwap_indicators_weight'] = 0.20
            data['greek_sentiment_weight'] = 0.25
            data['other_indicators_weight'] = 0.05
            return data
    
    def _classify_regimes(self, data):
        """
        Classify market regimes based on combined indicator component.
        
        Args:
            data (pd.DataFrame): Data with combined regime component.
            
        Returns:
            pd.DataFrame: Data with regime classifications.
        """
        try:
            # Initialize regime column
            data['regime'] = 'Unknown'
            
            # Classify directional bias
            data['directional_bias'] = 'Neutral'
            
            # Bullish conditions
            bullish = data['combined_regime_component'] > 0.3
            data.loc[bullish, 'directional_bias'] = 'Bullish'
            
            # Bearish conditions
            bearish = data['combined_regime_component'] < -0.3
            data.loc[bearish, 'directional_bias'] = 'Bearish'
            
            # Classify volatility level
            data['volatility_level'] = 'Medium_Vol'
            
            # Calculate volatility percentiles if volatility column exists
            if 'volatility' in data.columns:
                vol_low = data['volatility'].quantile(0.33)
                vol_high = data['volatility'].quantile(0.67)
                
                data.loc[data['volatility'] <= vol_low, 'volatility_level'] = 'Low_Vol'
                data.loc[data['volatility'] >= vol_high, 'volatility_level'] = 'High_Vol'
            else:
                # Use absolute value of combined component as a proxy for volatility
                component_abs = data['combined_regime_component'].abs()
                vol_low = component_abs.quantile(0.33)
                vol_high = component_abs.quantile(0.67)
                
                data.loc[component_abs <= vol_low, 'volatility_level'] = 'Low_Vol'
                data.loc[component_abs >= vol_high, 'volatility_level'] = 'High_Vol'
            
            # Classify skew pattern
            data['skew_pattern'] = 'Balanced'
            
            # Use vega component from Greek sentiment as a proxy for skew if available
            if 'Vega_Component' in data.columns:
                skew_low = data['Vega_Component'].quantile(0.33)
                skew_high = data['Vega_Component'].quantile(0.67)
                
                data.loc[data['Vega_Component'] <= skew_low, 'skew_pattern'] = 'Call_Skew'
                data.loc[data['Vega_Component'] >= skew_high, 'skew_pattern'] = 'Put_Skew'
            else:
                # Default to balanced skew
                pass
            
            # Combine classifications into regime
            data['regime'] = data['directional_bias'] + '_' + data['volatility_level'] + '_' + data['skew_pattern']
            
            # Drop intermediate columns to save memory
            if 'directional_bias' in data.columns:
                data = data.drop('directional_bias', axis=1)
            
            if 'volatility_level' in data.columns:
                data = data.drop('volatility_level', axis=1)
            
            if 'skew_pattern' in data.columns:
                data = data.drop('skew_pattern', axis=1)
            
            return data
        except Exception as e:
            logger.error(f"Error classifying regimes: {str(e)}")
            
            # Return original data with default regime
            data['regime'] = 'Neutral_Medium_Vol_Balanced'
            return data
    
    def _add_confidence_scores(self, data):
        """
        Add confidence scores to regime classifications.
        
        Args:
            data (pd.DataFrame): Data with regime classifications.
            
        Returns:
            pd.DataFrame: Data with confidence scores.
        """
        try:
            # Initialize confidence score
            data['confidence_score'] = 0.5  # Default medium confidence
            
            # Ensure all confidence columns exist with default values
            confidence_cols = {
                'trending_oi_pa_confidence': 0.5,
                'ema_confidence': 0.5,
                'vwap_confidence': 0.5,
                'greek_confidence': 0.5
            }
            
            for col, default_val in confidence_cols.items():
                if col not in data.columns:
                    data[col] = default_val
            
            # Combine confidence scores from individual indicators
            confidence_components = [
                data['trending_oi_pa_confidence'] * data['trending_oi_pa_weight'],
                data['ema_confidence'] * data['ema_indicators_weight'],
                data['vwap_confidence'] * data['vwap_indicators_weight'],
                data['greek_confidence'] * data['greek_sentiment_weight']
            ]
            
            # Calculate weighted average confidence
            weight_sum = (
                data['trending_oi_pa_weight'] +
                data['ema_indicators_weight'] +
                data['vwap_indicators_weight'] +
                data['greek_sentiment_weight']
            )
            
            data['confidence_score'] = sum(confidence_components) / weight_sum
            
            # Adjust confidence based on regime persistence
            # Higher confidence if regime has been stable for multiple minutes
            data['regime_shift'] = data['regime'] != data['regime'].shift(1)
            data['regime_duration'] = 1
            
            # Use a more memory-efficient approach for regime duration
            regime_duration = 1
            for i in range(1, len(data)):
                if data.iloc[i]['regime_shift']:
                    regime_duration = 1
                else:
                    regime_duration += 1
                data.iloc[i, data.columns.get_loc('regime_duration')] = regime_duration
            
            # Increase confidence for persistent regimes (up to 20% boost)
            data['persistence_boost'] = np.minimum(0.2, (data['regime_duration'] - 1) * 0.02)
            data['confidence_score'] = np.minimum(1.0, data['confidence_score'] + data['persistence_boost'])
            
            # Drop intermediate columns to save memory
            if 'regime_shift' in data.columns:
                data = data.drop('regime_shift', axis=1)
            
            if 'persistence_boost' in data.columns:
                data = data.drop('persistence_boost', axis=1)
            
            return data
        except Exception as e:
            logger.error(f"Error adding confidence scores: {str(e)}")
            
            # Return original data with default confidence score
            data['confidence_score'] = 0.5
            return data
    
    def _add_strategy_recommendations(self, data):
        """
        Add strategy recommendations based on regime classifications.
        
        Args:
            data (pd.DataFrame): Data with regime classifications.
            
        Returns:
            pd.DataFrame: Data with strategy recommendations.
        """
        try:
            # Initialize strategy recommendations
            data['recommended_strategies'] = None
            
            # Add recommendations based on regime
            for regime, strategies in self.regime_strategy_map.items():
                mask = data['regime'] == regime
                if mask.any():
                    data.loc[mask, 'recommended_strategies'] = str(strategies)
            
            return data
        except Exception as e:
            logger.error(f"Error adding strategy recommendations: {str(e)}")
            
            # Return original data with default recommendations
            data['recommended_strategies'] = "['long_call', 'long_put']"
            return data
    
    def run_analysis(self, file_paths):
        """
        Run analysis on multiple files and generate comprehensive report.
        
        Args:
            file_paths (list): List of file paths to analyze.
            
        Returns:
            dict: Analysis results and report.
        """
        logger.info(f"Running analysis on {len(file_paths)} files")
        self.memory_monitor.log_memory_usage("Before running analysis")
        
        all_results = []
        
        # Process each file
        for file_path in file_paths:
            result = self.process_file(file_path)
            if not result.empty:
                # Save result to file to save memory
                result_file = os.path.join(self.output_dir, f"result_{os.path.basename(file_path)}")
                result.to_csv(result_file, index=False)
                all_results.append(result_file)
                
                # Free memory
                del result
                gc.collect()
        
        # Check if we have any results
        if not all_results:
            logger.error("No results to combine")
            return {
                'results': pd.DataFrame(),
                'report': {},
                'output_dir': self.output_dir
            }
        
        # Combine results in batches to save memory
        logger.info("Combining results from multiple files")
        self.memory_monitor.log_memory_usage("Before combining results")
        
        batch_size = 2
        combined_results = []
        
        for i in range(0, len(all_results), batch_size):
            batch_files = all_results[i:i+batch_size]
            batch_dfs = []
            
            for file in batch_files:
                try:
                    df = pd.read_csv(file)
                    batch_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
            
            if batch_dfs:
                batch_combined = pd.concat(batch_dfs, ignore_index=True)
                
                # Save batch result
                batch_file = os.path.join(self.output_dir, f"batch_{i//batch_size+1}.csv")
                batch_combined.to_csv(batch_file, index=False)
                combined_results.append(batch_file)
                
                # Free memory
                del batch_dfs
                del batch_combined
                gc.collect()
        
        # Combine batch results
        final_dfs = []
        
        for file in combined_results:
            try:
                df = pd.read_csv(file)
                final_dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {file}: {str(e)}")
        
        if not final_dfs:
            logger.error("No final results to combine")
            return {
                'results': pd.DataFrame(),
                'report': {},
                'output_dir': self.output_dir
            }
        
        result = pd.concat(final_dfs, ignore_index=True)
        
        # Save combined results
        combined_file = os.path.join(self.output_dir, "combined_minute_regime_classification.csv")
        result.to_csv(combined_file, index=False)
        logger.info(f"Saved combined results to {combined_file}")
        self.memory_monitor.log_memory_usage("After saving combined results")
        
        # Generate report
        report = self._generate_report(result)
        
        # Save report
        report_file = os.path.join(self.output_dir, "minute_regime_analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved report to {report_file}")
        
        # Generate visualizations
        self._generate_visualizations(result)
        
        # Free memory
        del final_dfs
        del result
        gc.collect()
        
        return {
            'combined_file': combined_file,
            'report_file': report_file,
            'output_dir': self.output_dir
        }
    
    def _generate_report(self, data):
        """
        Generate comprehensive report from analysis results.
        
        Args:
            data (pd.DataFrame): Combined analysis results.
            
        Returns:
            dict: Report with key findings and statistics.
        """
        logger.info("Generating comprehensive report")
        self.memory_monitor.log_memory_usage("Before generating report")
        
        try:
            report = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_minutes_analyzed': len(data),
                'date_range': {
                    'start': data['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': data['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'regime_distribution': {},
                'confidence_statistics': {
                    'mean': float(data['confidence_score'].mean()),
                    'median': float(data['confidence_score'].median()),
                    'min': float(data['confidence_score'].min()),
                    'max': float(data['confidence_score'].max())
                },
                'indicator_weights': {
                    'trending_oi_pa': float(data['trending_oi_pa_weight'].iloc[-1]),
                    'ema_indicators': float(data['ema_indicators_weight'].iloc[-1]),
                    'vwap_indicators': float(data['vwap_indicators_weight'].iloc[-1]),
                    'greek_sentiment': float(data['greek_sentiment_weight'].iloc[-1]),
                    'other_indicators': float(data['other_indicators_weight'].iloc[-1])
                },
                'regime_persistence': {},
                'time_of_day_patterns': {},
                'indicator_contributions': {}
            }
            
            # Calculate regime distribution
            regime_counts = data['regime'].value_counts()
            total_regimes = len(data)
            
            for regime, count in regime_counts.items():
                report['regime_distribution'][regime] = {
                    'count': int(count),
                    'percentage': float(count / total_regimes * 100)
                }
            
            # Calculate regime persistence
            regime_durations = data.groupby('regime')['regime_duration'].max()
            
            for regime, duration in regime_durations.items():
                report['regime_persistence'][regime] = {
                    'max_duration_minutes': int(duration),
                    'avg_duration_minutes': float(data[data['regime'] == regime]['regime_duration'].mean())
                }
            
            # Calculate time-of-day patterns
            data['hour'] = pd.to_datetime(data['datetime']).dt.hour
            
            for hour in sorted(data['hour'].unique()):
                hour_data = data[data['hour'] == hour]
                hour_regime_counts = hour_data['regime'].value_counts()
                
                report['time_of_day_patterns'][str(hour)] = {
                    'top_regime': hour_regime_counts.index[0] if not hour_regime_counts.empty else 'Unknown',
                    'top_regime_percentage': float(hour_regime_counts.iloc[0] / len(hour_data) * 100) if not hour_regime_counts.empty else 0
                }
            
            # Calculate indicator contributions
            component_cols = {
                'trending_oi_pa': 'trending_oi_pa_component',
                'ema_indicators': 'ema_regime_component',
                'vwap_indicators': 'vwap_regime_component',
                'greek_sentiment': 'greek_regime_component'
            }
            
            for indicator, component_col in component_cols.items():
                if component_col in data.columns:
                    report['indicator_contributions'][indicator] = {
                        'mean_contribution': float(data[component_col].mean()),
                        'correlation_with_combined': float(data[component_col].corr(data['combined_regime_component'])),
                        'mean_weight': float(data[f"{indicator}_weight"].mean())
                    }
            
            # Get optimal weights from weight adjuster
            optimal_weights_report = self.weight_adjuster.get_optimal_weights_report()
            report['optimal_weights'] = optimal_weights_report
            
            logger.info("Report generation completed")
            self.memory_monitor.log_memory_usage("After generating report")
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            
            # Return basic report
            return {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'total_minutes_analyzed': len(data)
            }
    
    def _generate_visualizations(self, data):
        """
        Generate visualizations from analysis results.
        
        Args:
            data (pd.DataFrame): Combined analysis results.
        """
        logger.info("Generating visualizations")
        self.memory_monitor.log_memory_usage("Before generating visualizations")
        
        try:
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Regime distribution
            plt.figure(figsize=(12, 8))
            regime_counts = data['regime'].value_counts().head(10)  # Top 10 regimes
            sns.barplot(x=regime_counts.index, y=regime_counts.values)
            plt.title('Top 10 Market Regimes Distribution')
            plt.xlabel('Regime')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'regime_distribution.png'))
            plt.close()
            
            # 2. Confidence score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data['confidence_score'], bins=20, kde=True)
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'confidence_distribution.png'))
            plt.close()
            
            # 3. Regime persistence
            plt.figure(figsize=(12, 8))
            regime_durations = data.groupby('regime')['regime_duration'].max().sort_values(ascending=False).head(10)  # Top 10 regimes by duration
            sns.barplot(x=regime_durations.index, y=regime_durations.values)
            plt.title('Top 10 Regimes by Maximum Duration')
            plt.xlabel('Regime')
            plt.ylabel('Maximum Duration (minutes)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'regime_persistence.png'))
            plt.close()
            
            # 4. Time-of-day patterns
            plt.figure(figsize=(12, 8))
            data['hour'] = pd.to_datetime(data['datetime']).dt.hour
            hour_regime_counts = data.groupby(['hour', 'regime']).size().unstack().fillna(0)
            
            # Get top 5 regimes
            top_regimes = data['regime'].value_counts().head(5).index
            if not hour_regime_counts.empty and all(regime in hour_regime_counts.columns for regime in top_regimes):
                hour_regime_counts = hour_regime_counts[top_regimes]
                
                hour_regime_counts.plot(kind='bar', stacked=True)
                plt.title('Top 5 Regimes by Hour of Day')
                plt.xlabel('Hour')
                plt.ylabel('Count')
                plt.legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'time_of_day_patterns.png'))
            plt.close()
            
            # 5. Indicator contributions
            plt.figure(figsize=(10, 6))
            indicators = ['trending_oi_pa', 'ema_indicators', 'vwap_indicators', 'greek_sentiment']
            weights = [data[f"{indicator}_weight"].iloc[-1] for indicator in indicators]
            
            plt.pie(weights, labels=indicators, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Indicator Weights in Final Classification')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'indicator_weights.png'))
            plt.close()
            
            # 6. Regime transitions
            if len(data) > 1:
                # Use a more memory-efficient approach for regime transitions
                # Sample data to reduce memory usage
                sample_size = min(10000, len(data))
                sampled_data = data.sample(sample_size) if len(data) > sample_size else data
                
                plt.figure(figsize=(12, 8))
                sampled_data['next_regime'] = sampled_data['regime'].shift(-1)
                transitions = sampled_data.groupby(['regime', 'next_regime']).size().reset_index(name='count')
                transitions = transitions.sort_values('count', ascending=False).head(20)  # Top 20 transitions
                
                if not transitions.empty:
                    sns.barplot(x='regime', y='count', hue='next_regime', data=transitions)
                    plt.title('Top 20 Regime Transitions (Sampled Data)')
                    plt.xlabel('From Regime')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title='To Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'regime_transitions.png'))
                plt.close()
            
            logger.info(f"Visualizations saved to {viz_dir}")
            self.memory_monitor.log_memory_usage("After generating visualizations")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

def process_date_range(start_date, end_date):
    """
    Process data for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        dict: Analysis results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("minute_regime_classifier.log"),
            logging.StreamHandler()
        ]
    )
    
    # Install psutil if not already installed
    try:
        import psutil
    except ImportError:
        logger.info("Installing psutil package")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Initialize classifier with date filter
    config = {
        'output_dir': '/home/ubuntu/market_regime_testing/output/minute_regime_analysis',
        'chunk_size': 1000,  # Reduced chunk size for better memory efficiency
        'memory_threshold_mb': 1000,  # Memory threshold in MB
        'date_filter': {
            'start_date': start_date,
            'end_date': end_date
        }
    }
    
    classifier = MinuteRegimeClassifier(config)
    
    # Get data directory
    data_dir = '/home/ubuntu/market_regime_testing/data/market_data'
    
    # Get all CSV files
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        logger.error(f"No CSV files found in {data_dir}")
        return {}
    
    # Process files
    logger.info(f"Processing {len(all_files)} files for date range {start_date} to {end_date}")
    results = classifier.run_analysis(all_files)
    
    return results

# Main function for testing
def main():
    """
    Main function for testing the MinuteRegimeClassifier.
    """
    # Install psutil if not already installed
    try:
        import psutil
    except ImportError:
        print("Installing psutil package")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("minute_regime_classifier.log"),
            logging.StreamHandler()
        ]
    )
    
    # Process data for specific date range
    start_date = '2025-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    results = process_date_range(start_date, end_date)
    
    if results:
        logger.info(f"Analysis completed successfully")
        logger.info(f"Results saved to {results.get('output_dir')}")
        logger.info(f"Combined results file: {results.get('combined_file')}")
        logger.info(f"Report file: {results.get('report_file')}")
    else:
        logger.error("Analysis failed")

if __name__ == "__main__":
    main()
