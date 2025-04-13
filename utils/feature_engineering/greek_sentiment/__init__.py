"""
Greek Sentiment Analysis Module - Enhanced Implementation

This module implements Greek sentiment analysis for options trading based on the Greek_sentiment.md document,
analyzing Delta, Vega, Theta, and Gamma exposures across a range of strikes from 0.5 delta to 0.1 delta
for current week, next week, and current month expiries.

Features:
- Full delta range analysis (0.5 to 0.1 delta)
- Multiple expiry cycle support (current week, next week, monthly)
- Baseline vs. current Greek comparison
- Weighted sentiment score calculation
- Expiry cycle weightage (70% near, 30% next)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from ..base import FeatureBase, register_feature, cache_result, time_execution

logger = logging.getLogger(__name__)

@register_feature(name="GreekSentiment", category="greek_sentiment")
class GreekSentiment(FeatureBase):
    """
    Enhanced Greek sentiment analysis for options trading.
    
    This class analyzes options Greeks (Delta, Gamma, Theta, Vega) across multiple
    strikes (0.5 delta to 0.1 delta) and expiry cycles to determine market sentiment.
    """
    
    def __init__(self, config=None):
        """
        Initialize Greek sentiment analysis.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config = config or {}
        
        # Core parameters
        try:
            self.lookback_period = int(self.config.get('lookback_period', 20))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert lookback_period to int, using default")
            self.lookback_period = 20
            
        try:
            # Remove any comments before conversion
            gamma_threshold_str = str(self.config.get('gamma_threshold', 0.7))
            if ';' in gamma_threshold_str:
                gamma_threshold_str = gamma_threshold_str.split(';')[0].strip()
            self.gamma_threshold = float(gamma_threshold_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert gamma_threshold to float, using default")
            self.gamma_threshold = 0.7
            
        try:
            delta_threshold_str = str(self.config.get('delta_threshold', 0.6))
            if ';' in delta_threshold_str:
                delta_threshold_str = delta_threshold_str.split(';')[0].strip()
            self.delta_threshold = float(delta_threshold_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert delta_threshold to float, using default")
            self.delta_threshold = 0.6
            
        try:
            sentiment_weight_str = str(self.config.get('sentiment_weight', 0.4))
            if ';' in sentiment_weight_str:
                sentiment_weight_str = sentiment_weight_str.split(';')[0].strip()
            self.sentiment_weight = float(sentiment_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert sentiment_weight to float, using default")
            self.sentiment_weight = 0.4
        
        # Expiry weights
        try:
            current_week_weight_str = str(self.config.get('current_week_weight', 0.7))
            if ';' in current_week_weight_str:
                current_week_weight_str = current_week_weight_str.split(';')[0].strip()
            self.current_week_weight = float(current_week_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert current_week_weight to float, using default")
            self.current_week_weight = 0.7
            
        try:
            next_week_weight_str = str(self.config.get('next_week_weight', 0.2))
            if ';' in next_week_weight_str:
                next_week_weight_str = next_week_weight_str.split(';')[0].strip()
            self.next_week_weight = float(next_week_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert next_week_weight to float, using default")
            self.next_week_weight = 0.2
            
        try:
            current_month_weight_str = str(self.config.get('current_month_weight', 0.1))
            if ';' in current_month_weight_str:
                current_month_weight_str = current_month_weight_str.split(';')[0].strip()
            self.current_month_weight = float(current_month_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert current_month_weight to float, using default")
            self.current_month_weight = 0.1
        
        # Greek weights
        try:
            vega_weight_str = str(self.config.get('vega_weight', 1.0))
            if ';' in vega_weight_str:
                vega_weight_str = vega_weight_str.split(';')[0].strip()
            self.vega_weight = float(vega_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert vega_weight to float, using default")
            self.vega_weight = 1.0
            
        try:
            delta_weight_str = str(self.config.get('delta_weight', 1.0))
            if ';' in delta_weight_str:
                delta_weight_str = delta_weight_str.split(';')[0].strip()
            self.delta_weight = float(delta_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert delta_weight to float, using default")
            self.delta_weight = 1.0
            
        try:
            theta_weight_str = str(self.config.get('theta_weight', 0.5))
            if ';' in theta_weight_str:
                theta_weight_str = theta_weight_str.split(';')[0].strip()
            self.theta_weight = float(theta_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert theta_weight to float, using default")
            self.theta_weight = 0.5
            
        try:
            gamma_weight_str = str(self.config.get('gamma_weight', 0.3))
            if ';' in gamma_weight_str:
                gamma_weight_str = gamma_weight_str.split(';')[0].strip()
            self.gamma_weight = float(gamma_weight_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert gamma_weight to float, using default")
            self.gamma_weight = 0.3
        
        # Delta range for analysis (from ATM to OTM)
        try:
            max_delta_str = str(self.config.get('max_delta', 0.5))
            if ';' in max_delta_str:
                max_delta_str = max_delta_str.split(';')[0].strip()
            self.max_delta = float(max_delta_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert max_delta to float, using default")
            self.max_delta = 0.5
            
        try:
            min_delta_str = str(self.config.get('min_delta', 0.1))
            if ';' in min_delta_str:
                min_delta_str = min_delta_str.split(';')[0].strip()
            self.min_delta = float(min_delta_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert min_delta to float, using default")
            self.min_delta = 0.1
        
        # Feature toggles
        use_synthetic_data_val = str(self.config.get('use_synthetic_data', True)).lower()
        self.use_synthetic_data = use_synthetic_data_val in ['true', '1', 'yes', 'y']
        
        use_baseline_comparison_val = str(self.config.get('use_baseline_comparison', True)).lower()
        self.use_baseline_comparison = use_baseline_comparison_val in ['true', '1', 'yes', 'y']
        
        logger.info(f"Initialized enhanced Greek sentiment analysis")
        logger.info(f"Delta range: {self.max_delta} to {self.min_delta}")
        logger.info(f"Expiry weights: current_week={self.current_week_weight}, next_week={self.next_week_weight}, current_month={self.current_month_weight}")
    
    """
    Implementation Plan:
    
    1. calculate_features method:
       - Parse input data frames with columns for different strikes and expiries
       - Separate data by expiry cycles
       - Compute baseline Greeks (from market open)
       - Calculate Greek changes for each strike in the delta range
       - Aggregate changes across strikes for calls and puts
       - Calculate sentiment score using the formula from the document
       - Apply expiry weightages to get the final sentiment
       
    2. Helper methods:
       - process_expiry_data: Process a single expiry cycle's data
       - calculate_greek_changes: Calculate changes from baseline for each Greek
       - aggregate_strikes: Sum the changes across strike range
       - calculate_sentiment: Interpret the aggregated changes as sentiment
       - combine_expiry_sentiment: Apply weights to combine sentiment from different expiries
       
    3. Synthetic data generation:
       - Enhanced to generate data for a range of strikes and multiple expiries
    """
    
    @cache_result
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate Greek sentiment features based on the Greek_sentiment.md document.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - baseline_time (str): Time to use as baseline (e.g., '09:15:00')
                - expiry_column (str): Column name for expiry information
                - delta_column_pattern (str): Pattern for delta columns (e.g., '{side}_{strike}_{greek}')
                - side_values (dict): Dictionary mapping sides (e.g., 'call', 'put')
                
        Returns:
            pd.DataFrame: Data with calculated Greek sentiment features
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Parse kwargs
        baseline_time = kwargs.get('baseline_time', '09:15:00')
        expiry_column = kwargs.get('expiry_column', 'Expiry')
        
        # Log parameters
        logger.info(f"Calculating Greek sentiment with baseline time: {baseline_time}")
        
        # Check if required columns exist
        if 'Time' not in df.columns:
            logger.warning("Time column not found, using first row as baseline")
            baseline_rows = df.iloc[:1]
        else:
            # Find baseline rows (market open)
            baseline_rows = df[df['Time'] == baseline_time] if baseline_time in df['Time'].values else df.iloc[:1]
            if len(baseline_rows) == 0:
                logger.warning(f"Baseline time {baseline_time} not found, using first row as baseline")
                baseline_rows = df.iloc[:1]
        
        # Check for expiry column
        if expiry_column not in df.columns:
            if self.use_synthetic_data:
                logger.info(f"Adding synthetic expiry column '{expiry_column}'")
                # Create a more realistic distribution of expiry cycles
                expiry_values = np.random.choice(
                    ['current_week', 'next_week', 'current_month'], 
                    size=len(df),
                    p=[0.6, 0.3, 0.1]  # More weight to current week
                )
                df[expiry_column] = expiry_values
            else:
                logger.error(f"Expiry column '{expiry_column}' not found and synthetic data disabled")
                return df
        
        # Identify Greek columns for each strike in the delta range
        greek_columns = self._identify_greek_columns(df, **kwargs)
        
        if not greek_columns:
            if self.use_synthetic_data:
                logger.warning("No Greek columns found, generating synthetic data")
                df = self._generate_synthetic_greek_data(df, expiry_column)
                greek_columns = self._identify_greek_columns(df, **kwargs)
            else:
                logger.error("No Greek columns found and synthetic data disabled")
                return df
        
        # Split data by expiry cycle
        current_week_data = df[df[expiry_column] == 'current_week'].copy() if 'current_week' in df[expiry_column].values else None
        next_week_data = df[df[expiry_column] == 'next_week'].copy() if 'next_week' in df[expiry_column].values else None
        current_month_data = df[df[expiry_column] == 'current_month'].copy() if 'current_month' in df[expiry_column].values else None
        
        # Process each expiry cycle
        sentiment_by_expiry = {}
        
        if current_week_data is not None and len(current_week_data) > 0:
            current_week_baseline = baseline_rows[baseline_rows[expiry_column] == 'current_week'] if expiry_column in baseline_rows.columns else baseline_rows
            sentiment_by_expiry['current_week'] = self._process_expiry_data(
                current_week_data, current_week_baseline, greek_columns, 'current_week'
            )
        
        if next_week_data is not None and len(next_week_data) > 0:
            next_week_baseline = baseline_rows[baseline_rows[expiry_column] == 'next_week'] if expiry_column in baseline_rows.columns else baseline_rows
            sentiment_by_expiry['next_week'] = self._process_expiry_data(
                next_week_data, next_week_baseline, greek_columns, 'next_week'
            )
        
        if current_month_data is not None and len(current_month_data) > 0:
            current_month_baseline = baseline_rows[baseline_rows[expiry_column] == 'current_month'] if expiry_column in baseline_rows.columns else baseline_rows
            sentiment_by_expiry['current_month'] = self._process_expiry_data(
                current_month_data, current_month_baseline, greek_columns, 'current_month'
            )
        
        # Combine sentiment from different expiry cycles
        combined_sentiment = self._combine_expiry_sentiment(sentiment_by_expiry)
        
        # Add combined sentiment to original dataframe
        for col in ['Greek_Sentiment', 'Greek_Sentiment_Regime', 'SumVegaCall', 'SumVegaPut', 
                    'SumDeltaCall', 'SumDeltaPut', 'SumThetaCall', 'SumThetaPut']:
            if col in combined_sentiment.columns:
                df[col] = np.nan
                
                # Ensure Date column exists and is in the right format
                if 'Date' not in df.columns:
                    logger.warning("Date column not found in dataframe, creating a dummy date column")
                    df['Date'] = pd.Timestamp.now().date()
                if 'Date' not in combined_sentiment.columns:
                    logger.warning("Date column not found in combined_sentiment, creating a dummy date column")
                    combined_sentiment['Date'] = pd.Timestamp.now().date()
                
                # Convert Date columns to same type if they aren't already
                try:
                    if not pd.api.types.is_dtype_equal(df['Date'].dtype, combined_sentiment['Date'].dtype):
                        logger.info("Converting Date columns to compatible format")
                        df['Date'] = pd.to_datetime(df['Date']).dt.date
                        combined_sentiment['Date'] = pd.to_datetime(combined_sentiment['Date']).dt.date
                except Exception as e:
                    logger.warning(f"Error converting Date columns: {str(e)}, using dummy dates")
                    # Create a temporary mapping by index since dates don't match
                    df['_temp_idx'] = np.arange(len(df))
                    combined_dates = combined_sentiment['Date'].unique()
                    
                    for i, date_val in enumerate(combined_dates):
                        # Create index-based mask instead of date-based 
                        start_idx = (i * len(df)) // len(combined_dates)
                        end_idx = ((i + 1) * len(df)) // len(combined_dates)
                        idx_mask = (df['_temp_idx'] >= start_idx) & (df['_temp_idx'] < end_idx)
                        
                        # Assign values based on index mask
                        source_val = combined_sentiment[combined_sentiment['Date'] == date_val][col].values
                        if len(source_val) > 0:
                            if col in ['Greek_Sentiment_Regime'] and isinstance(source_val[0], str):
                                try:
                                    df.loc[idx_mask, col] = int(source_val[0])
                                except (ValueError, TypeError):
                                    regime_map = {'Neutral': 0, 'Bullish': 1, 'Very Bullish': 2, 
                                                 'Bearish': -1, 'Very Bearish': -2}
                                    if source_val[0] in regime_map:
                                        df.loc[idx_mask, col] = regime_map[source_val[0]]
                                    else:
                                        df.loc[idx_mask, col] = 0
                            elif col == 'Greek_Sentiment' and isinstance(source_val[0], str):
                                try:
                                    df.loc[idx_mask, col] = float(source_val[0])
                                except (ValueError, TypeError):
                                    df.loc[idx_mask, col] = 0.0
                            else:
                                df.loc[idx_mask, col] = source_val[0]
                    
                    # Drop temporary index
                    df.drop('_temp_idx', axis=1, inplace=True)
                    continue
                
                # Standard date-based assignment if date conversion worked
                for date_val in combined_sentiment['Date'].unique():
                    try:
                        date_mask = df['Date'] == date_val
                        # Ensure values are converted to the correct type before assignment
                        source_val = combined_sentiment[combined_sentiment['Date'] == date_val][col].values
                        if len(source_val) > 0:
                            if col in ['Greek_Sentiment_Regime'] and isinstance(source_val[0], str):
                                # Convert string regime values to integers
                                try:
                                    df.loc[date_mask, col] = int(source_val[0])
                                except (ValueError, TypeError):
                                    # If conversion fails, map common words to values
                                    regime_map = {'Neutral': 0, 'Bullish': 1, 'Very Bullish': 2, 
                                                'Bearish': -1, 'Very Bearish': -2}
                                    if source_val[0] in regime_map:
                                        df.loc[date_mask, col] = regime_map[source_val[0]]
                                    else:
                                        df.loc[date_mask, col] = 0
                            elif col == 'Greek_Sentiment' and isinstance(source_val[0], str):
                                # Convert string sentiment values to float
                                try:
                                    df.loc[date_mask, col] = float(source_val[0])
                                except (ValueError, TypeError):
                                    # Default to neutral sentiment
                                    df.loc[date_mask, col] = 0.0
                            else:
                                df.loc[date_mask, col] = source_val[0]
                    except Exception as e:
                        logger.warning(f"Error assigning values for date {date_val}: {str(e)}")
                        # Skip this date and continue with others
                        continue
        
        return df
    
    def _identify_greek_columns(self, data_frame, **kwargs):
        """
        Identify Greek columns for different strikes in the data.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - delta_column_pattern (str): Pattern for delta columns
                
        Returns:
            dict: Dictionary of identified Greek columns by strike and type
        """
        # Default pattern matches columns like "Call_0.5_Delta", "Put_0.3_Vega", etc.
        default_pattern = "{side}_{delta}_"
        pattern = kwargs.get('delta_column_pattern', default_pattern)
        
        # Default side values
        default_sides = {'Call': 'Call', 'Put': 'Put'}
        sides = kwargs.get('side_values', default_sides)
        
        # Default Greek types to look for
        greek_types = kwargs.get('greek_types', ['Delta', 'Vega', 'Theta', 'Gamma'])
        
        # Find columns that match the pattern
        greek_columns = {}
        
        # Log all columns for debugging
        logger.info(f"Available columns: {', '.join(data_frame.columns)}")
        
        # If pattern uses {delta}, try to find columns for specific delta values
        if '{delta}' in pattern:
            # Generate potential delta values in the range
            delta_values = np.round(np.arange(self.min_delta, self.max_delta + 0.1, 0.1), 1)
            
            for side_name, side_value in sides.items():
                greek_columns[side_name] = {}
                
                for delta in delta_values:
                    greek_columns[side_name][str(delta)] = {}
                    
                    for greek in greek_types:
                        col_pattern = pattern.format(side=side_value, delta=delta) + greek
                        matching_cols = [col for col in data_frame.columns if col_pattern in col]
                        
                        if matching_cols:
                            greek_columns[side_name][str(delta)][greek] = matching_cols[0]
        
        # Try multiple pattern matching approaches if no columns found
        if not any(greek_columns.values()):
            logger.warning(f"No columns found with pattern '{pattern}', trying general search")
            
            for side_name, side_value in sides.items():
                greek_columns[side_name] = {}
                
                # General approach 1: Look for any columns containing the side name and Greek type
                for greek in greek_types:
                    matching_cols = [col for col in data_frame.columns 
                                    if side_value in col and greek in col]
                    
                    # If we found something, try to identify strikes from the column names
                    if matching_cols:
                        for col in matching_cols:
                            # Try to extract strike from column name
                            parts = col.split('_')
                            strike = None
                            
                            # Look for a part that might be a strike value
                            for part in parts:
                                try:
                                    # Try to convert to float to see if it's a strike value
                                    val = float(part)
                                    if 0 < val <= 1:  # Delta values are between 0 and 1
                                        strike = part
                                        break
                                except ValueError:
                                    continue
                            
                            # If we couldn't find a strike, use the column name as the key
                            if not strike:
                                strike = col.replace(side_value, '').replace(greek, '').strip('_')
                            
                            if strike not in greek_columns[side_name]:
                                greek_columns[side_name][strike] = {}
                            greek_columns[side_name][strike][greek] = col
            
            # General approach 2: Try common patterns if still nothing found
            if not any(greek_columns.values()):
                common_patterns = [
                    "{side}{delta}{greek}",  # CallDelta50
                    "{side}.{delta}.{greek}",  # Call.50.Delta
                    "{side}{greek}{delta}",  # CallDelta50
                    "{greek}_{side}_{delta}"  # Delta_Call_50
                ]
                
                for test_pattern in common_patterns:
                    logger.info(f"Trying pattern: {test_pattern}")
                    # Skip this iteration if we already found columns
                    if any(greek_columns.values()):
                        break
                        
                    for side_name, side_value in sides.items():
                        if side_name not in greek_columns:
                            greek_columns[side_name] = {}
                            
                        for delta in delta_values:
                            for greek in greek_types:
                                for delta_str in [str(delta), str(int(delta*100))]:
                                    test_col = test_pattern.format(
                                        side=side_value, 
                                        delta=delta_str,
                                        greek=greek
                                    )
                                    matching_cols = [col for col in data_frame.columns 
                                                    if test_col.lower() in col.lower()]
                                    
                                    if matching_cols:
                                        if str(delta) not in greek_columns[side_name]:
                                            greek_columns[side_name][str(delta)] = {}
                                        greek_columns[side_name][str(delta)][greek] = matching_cols[0]
        
        # Count found columns for logging
        found_columns = sum(1 for side in greek_columns for delta in greek_columns[side]
                           for greek in greek_columns[side][delta] if greek_columns[side][delta][greek])
        
        logger.info(f"Identified {found_columns} Greek columns across {len(greek_columns['Call']) if 'Call' in greek_columns else 0} call strikes and {len(greek_columns['Put']) if 'Put' in greek_columns else 0} put strikes")
        
        return greek_columns
    
    def _process_expiry_data(self, expiry_data, baseline_rows, greek_columns, expiry_name):
        """
        Process data for a single expiry cycle.
        
        Args:
            expiry_data (DataFrame): Data for the expiry cycle
            baseline_rows (DataFrame): Baseline data (usually market open)
            greek_columns (dict): Dictionary of identified Greek columns
            expiry_name (str): Name of the expiry cycle
            
        Returns:
            DataFrame: Sentiment data for the expiry cycle
        """
        logger.info(f"Processing {expiry_name} data with {len(expiry_data)} rows")
        
        # Check if we have data
        if expiry_data is None or len(expiry_data) == 0:
            logger.warning(f"No data available for {expiry_name}")
            return None
            
        # Check if we have baseline data
        if baseline_rows is None or len(baseline_rows) == 0:
            logger.warning(f"No baseline data for {expiry_name}, using first row as baseline")
            baseline_rows = expiry_data.iloc[:1]
        
        # Ensure Date column exists and is in the right format
        if 'Date' not in expiry_data.columns:
            logger.warning(f"Date column not found in {expiry_name} data, creating dummy dates")
            expiry_data['Date'] = pd.Timestamp.now().date()
        else:
            # Convert to datetime.date
            try:
                expiry_data['Date'] = pd.to_datetime(expiry_data['Date']).dt.date
            except Exception as e:
                logger.warning(f"Error converting Date column in {expiry_name}: {str(e)}")
                # Keep existing Date column
        
        try:
            # Calculate Greek changes for each strike
            call_changes = self._calculate_greek_changes(
                expiry_data, baseline_rows, greek_columns, 'Call'
            )
            
            put_changes = self._calculate_greek_changes(
                expiry_data, baseline_rows, greek_columns, 'Put'
            )
            
            # Get aggregate strike changes
            aggregated_changes = self._aggregate_strikes(call_changes, put_changes)
            
            # Calculate sentiment
            sentiment = self._calculate_sentiment(aggregated_changes)
            
            # Set expiry name
            sentiment['Expiry'] = expiry_name
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error processing {expiry_name} data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a minimal valid dataframe
            unique_dates = expiry_data['Date'].unique()
            if len(unique_dates) == 0:
                unique_dates = [pd.Timestamp.now().date()]
                
            # Create a minimal sentiment dataframe with neutral values
            result = pd.DataFrame({
                'Date': unique_dates,
                'Greek_Sentiment': 0.0,
                'Greek_Sentiment_Regime': 0,
                'SumVegaCall': 0.0,
                'SumVegaPut': 0.0,
                'SumDeltaCall': 0.0,
                'SumDeltaPut': 0.0,
                'SumThetaCall': 0.0,
                'SumThetaPut': 0.0,
                'Expiry': expiry_name
            })
            
            return result
    
    def _calculate_greek_changes(self, data, baseline, greek_columns, side):
        """
        Calculate changes in Greeks from baseline for each strike.
        
        Args:
            data (DataFrame): Current data
            baseline (DataFrame): Baseline data (market open)
            greek_columns (dict): Dictionary of identified Greek columns
            side (str): 'Call' or 'Put'
            
        Returns:
            dict: Dictionary of Greek changes by strike and type
        """
        changes = {}
        
        # Process each delta/strike
        for delta, greeks in greek_columns[side].items():
            changes[delta] = {}
            
            # Process each Greek type
            for greek_type, column in greeks.items():
                if column in data.columns and column in baseline.columns:
                    # Calculate change from baseline
                    baseline_value = baseline[column].iloc[0]
                    current_values = data[column]
                    
                    # Store the change for each timestamp
                    changes[delta][greek_type] = current_values - baseline_value
                    
                    # For tracking the direction of change
                    changes[delta][f'{greek_type}_direction'] = np.sign(changes[delta][greek_type])
        
        return changes
    
    def _aggregate_strikes(self, call_changes, put_changes):
        """
        Aggregate changes across strikes for calls and puts.
        
        Args:
            call_changes (dict): Changes in call Greeks by strike
            put_changes (dict): Changes in put Greeks by strike
            
        Returns:
            dict: Aggregated changes
        """
        # Initialize aggregated changes
        aggregated = {
            'SumVegaCall': 0,
            'SumVegaPut': 0,
            'SumDeltaCall': 0,
            'SumDeltaPut': 0,
            'SumThetaCall': 0,
            'SumThetaPut': 0,
            'SumGammaCall': 0,
            'SumGammaPut': 0
        }
        
        # Aggregate call changes
        for delta, greeks in call_changes.items():
            if 'Vega' in greeks:
                aggregated['SumVegaCall'] += greeks['Vega'].sum() if isinstance(greeks['Vega'], pd.Series) else greeks['Vega']
            
            if 'Delta' in greeks:
                aggregated['SumDeltaCall'] += greeks['Delta'].sum() if isinstance(greeks['Delta'], pd.Series) else greeks['Delta']
            
            if 'Theta' in greeks:
                aggregated['SumThetaCall'] += greeks['Theta'].sum() if isinstance(greeks['Theta'], pd.Series) else greeks['Theta']
            
            if 'Gamma' in greeks:
                aggregated['SumGammaCall'] += greeks['Gamma'].sum() if isinstance(greeks['Gamma'], pd.Series) else greeks['Gamma']
        
        # Aggregate put changes
        for delta, greeks in put_changes.items():
            if 'Vega' in greeks:
                aggregated['SumVegaPut'] += greeks['Vega'].sum() if isinstance(greeks['Vega'], pd.Series) else greeks['Vega']
            
            if 'Delta' in greeks:
                aggregated['SumDeltaPut'] += greeks['Delta'].sum() if isinstance(greeks['Delta'], pd.Series) else greeks['Delta']
            
            if 'Theta' in greeks:
                aggregated['SumThetaPut'] += greeks['Theta'].sum() if isinstance(greeks['Theta'], pd.Series) else greeks['Theta']
            
            if 'Gamma' in greeks:
                aggregated['SumGammaPut'] += greeks['Gamma'].sum() if isinstance(greeks['Gamma'], pd.Series) else greeks['Gamma']
        
        return aggregated
    
    def _calculate_sentiment(self, data):
        """
        Calculate Greek sentiment from aggregated Greek changes.
        
        Args:
            data (DataFrame or dict): Aggregated Greek data
            
        Returns:
            DataFrame: Sentiment data with Date and sentiment columns
        """
        try:
            # Convert dict to DataFrame if necessary
            if isinstance(data, dict):
                # Check if data contains the expected keys
                expected_keys = ['Date', 'SumVegaCall', 'SumVegaPut', 'SumDeltaCall', 'SumDeltaPut']
                missing_keys = [key for key in expected_keys if key not in data]
                if missing_keys:
                    logger.warning(f"Missing keys in input data: {missing_keys}")
                    # Create a default dict with required keys
                    default_length = max([len(data[k]) for k in data if hasattr(data[k], '__len__')], default=1)
                    for key in missing_keys:
                        if key == 'Date':
                            data[key] = [pd.Timestamp.now().date()] * default_length
                        else:
                            data[key] = [0.0] * default_length
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Initialize result DataFrame
            result_columns = ['Date', 'Greek_Sentiment', 'Greek_Sentiment_Regime']
            if 'Date' in df.columns:
                result = pd.DataFrame({'Date': df['Date'].unique()})
            else:
                logger.warning("Date column not found in input data, creating dummy dates")
                result = pd.DataFrame({'Date': [pd.Timestamp.now().date()]})
                
            # Initialize sentiment columns
            result['Greek_Sentiment'] = 0.0
            result['Greek_Sentiment_Regime'] = 0
            
            # Calculate vega and delta exposure
            vega_call = df['SumVegaCall'] if 'SumVegaCall' in df.columns else 0
            vega_put = df['SumVegaPut'] if 'SumVegaPut' in df.columns else 0
            delta_call = df['SumDeltaCall'] if 'SumDeltaCall' in df.columns else 0
            delta_put = df['SumDeltaPut'] if 'SumDeltaPut' in df.columns else 0
            
            # Calculate net exposures
            net_vega = vega_call - vega_put
            net_delta = delta_call + delta_put  # Put delta is negative by convention
            
            # Normalize to [-1, 1] range
            max_vega = max(1, abs(net_vega).max() if hasattr(net_vega, 'max') else abs(net_vega))
            max_delta = max(1, abs(net_delta).max() if hasattr(net_delta, 'max') else abs(net_delta))
            
            vega_sentiment = net_vega / max_vega
            delta_sentiment = net_delta / max_delta
            
            # Apply weights to get the final sentiment
            sentiment = (
                self.vega_weight * vega_sentiment + 
                self.delta_weight * delta_sentiment
            ) / (self.vega_weight + self.delta_weight)
            
            # Ensure sentiment is in [-1, 1] range
            sentiment = np.clip(sentiment, -1, 1)
            
            # Convert to sentiment regime (-2 to +2)
            regime = np.zeros_like(sentiment) if hasattr(sentiment, 'shape') else 0
            
            if hasattr(sentiment, '__iter__'):
                regime = np.zeros_like(sentiment)
                regime[sentiment > 0.2] = 1    # Bullish
                regime[sentiment < -0.2] = -1  # Bearish
                regime[sentiment > 0.6] = 2    # Very Bullish
                regime[sentiment < -0.6] = -2  # Very Bearish
            else:
                if sentiment > 0.2:
                    regime = 1
                elif sentiment < -0.2:
                    regime = -1
                if sentiment > 0.6:
                    regime = 2
                elif sentiment < -0.6:
                    regime = -2
            
            # Assign to result DataFrame
            # Handle both Series and scalar values
            if hasattr(sentiment, '__iter__') and len(sentiment) == len(result):
                result['Greek_Sentiment'] = sentiment
                result['Greek_Sentiment_Regime'] = regime
            else:
                result['Greek_Sentiment'] = sentiment
                result['Greek_Sentiment_Regime'] = regime
            
            # Add original aggregated data to result
            for col in ['SumVegaCall', 'SumVegaPut', 'SumDeltaCall', 'SumDeltaPut', 'SumThetaCall', 'SumThetaPut']:
                if col in df.columns:
                    if len(df[col]) == len(result):
                        result[col] = df[col].values
                    else:
                        # If lengths don't match, fill with the first value or zero
                        result[col] = df[col].iloc[0] if len(df[col]) > 0 else 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a minimal valid result
            if isinstance(data, dict) and 'Date' in data and hasattr(data['Date'], '__iter__'):
                dates = data['Date']
            elif isinstance(data, pd.DataFrame) and 'Date' in data.columns:
                dates = data['Date'].unique()
            else:
                dates = [pd.Timestamp.now().date()]
                
            result = pd.DataFrame({
                'Date': dates,
                'Greek_Sentiment': 0.0,
                'Greek_Sentiment_Regime': 0
            })
            
            return result
    
    def _combine_expiry_sentiment(self, sentiment_by_expiry):
        """
        Combine sentiment from different expiry cycles using expiry weights.
        
        Args:
            sentiment_by_expiry (dict): Dictionary of sentiment data by expiry
            
        Returns:
            DataFrame: Combined sentiment data
        """
        if not sentiment_by_expiry:
            logger.warning("No sentiment data by expiry cycle available")
            # Return a dummy dataframe with minimal columns
            result = pd.DataFrame({
                'Date': [pd.Timestamp.now().date()],
                'Greek_Sentiment': [0.0],
                'Greek_Sentiment_Regime': [0]
            })
            return result
        
        # Get all unique dates across all expiry cycles
        all_dates = set()
        for expiry, data in sentiment_by_expiry.items():
            if data is None or not isinstance(data, pd.DataFrame) or len(data) == 0:
                logger.warning(f"Empty or invalid data for expiry cycle: {expiry}")
                continue
                
            if 'Date' not in data.columns:
                logger.warning(f"Date column missing in {expiry} expiry data")
                continue
                
            # Ensure Date is in the right format
            try:
                data['Date'] = pd.to_datetime(data['Date']).dt.date
                all_dates.update(data['Date'].unique())
            except Exception as e:
                logger.warning(f"Error processing dates for {expiry} expiry: {str(e)}")
        
        if not all_dates:
            logger.warning("No valid dates found across expiry cycles")
            result = pd.DataFrame({
                'Date': [pd.Timestamp.now().date()],
                'Greek_Sentiment': [0.0],
                'Greek_Sentiment_Regime': [0]
            })
            return result
            
        # Create combined dataframe with all dates
        combined = pd.DataFrame({'Date': list(all_dates)})
        
        # Add columns for sentiment and regime
        combined['Greek_Sentiment'] = 0.0
        combined['Greek_Sentiment_Regime'] = 0
        
        # Add columns for Greek aggregates
        for col in ['SumVegaCall', 'SumVegaPut', 'SumDeltaCall', 'SumDeltaPut', 
                   'SumThetaCall', 'SumThetaPut']:
            combined[col] = 0.0
        
        # Process each expiry cycle
        try:
            # Process current week if available
            if 'current_week' in sentiment_by_expiry and sentiment_by_expiry['current_week'] is not None:
                current_week_data = sentiment_by_expiry['current_week']
                if not isinstance(current_week_data, pd.DataFrame) or len(current_week_data) == 0:
                    logger.warning("Empty or invalid data for current week")
                else:
                    # Ensure Date is in the right format
                    try:
                        current_week_data['Date'] = pd.to_datetime(current_week_data['Date']).dt.date
                    except Exception as e:
                        logger.warning(f"Error converting dates for current week: {str(e)}")
                        current_week_data['Date'] = pd.Timestamp.now().date()
                    
                    for date in current_week_data['Date'].unique():
                        try:
                            date_data = current_week_data[current_week_data['Date'] == date]
                            date_mask = combined['Date'] == date
                            
                            if not date_mask.any():
                                logger.debug(f"Date {date} not found in combined data")
                                continue
                            
                            for col in ['Greek_Sentiment', 'Greek_Sentiment_Regime', 'SumVegaCall', 'SumVegaPut',
                                       'SumDeltaCall', 'SumDeltaPut', 'SumThetaCall', 'SumThetaPut']:
                                if col in date_data.columns and len(date_data[col]) > 0:
                                    weight = self.current_week_weight
                                    combined.loc[date_mask, col] += weight * date_data[col].values[0]
                        except Exception as e:
                            logger.warning(f"Error processing current week date {date}: {str(e)}")
                            continue
            
            # Process next week if available
            if 'next_week' in sentiment_by_expiry and sentiment_by_expiry['next_week'] is not None:
                next_week_data = sentiment_by_expiry['next_week']
                if not isinstance(next_week_data, pd.DataFrame) or len(next_week_data) == 0:
                    logger.warning("Empty or invalid data for next week")
                else:
                    # Ensure Date is in the right format
                    try:
                        next_week_data['Date'] = pd.to_datetime(next_week_data['Date']).dt.date
                    except Exception as e:
                        logger.warning(f"Error converting dates for next week: {str(e)}")
                        next_week_data['Date'] = pd.Timestamp.now().date()
                    
                    for date in next_week_data['Date'].unique():
                        try:
                            date_data = next_week_data[next_week_data['Date'] == date]
                            date_mask = combined['Date'] == date
                            
                            if not date_mask.any():
                                logger.debug(f"Date {date} not found in combined data")
                                continue
                            
                            for col in ['Greek_Sentiment', 'Greek_Sentiment_Regime', 'SumVegaCall', 'SumVegaPut',
                                       'SumDeltaCall', 'SumDeltaPut', 'SumThetaCall', 'SumThetaPut']:
                                if col in date_data.columns and len(date_data[col]) > 0:
                                    weight = self.next_week_weight
                                    combined.loc[date_mask, col] += weight * date_data[col].values[0]
                        except Exception as e:
                            logger.warning(f"Error processing next week date {date}: {str(e)}")
                            continue
            
            # Process current month if available
            if 'current_month' in sentiment_by_expiry and sentiment_by_expiry['current_month'] is not None:
                current_month_data = sentiment_by_expiry['current_month']
                if not isinstance(current_month_data, pd.DataFrame) or len(current_month_data) == 0:
                    logger.warning("Empty or invalid data for current month")
                else:
                    # Ensure Date is in the right format
                    try:
                        current_month_data['Date'] = pd.to_datetime(current_month_data['Date']).dt.date
                    except Exception as e:
                        logger.warning(f"Error converting dates for current month: {str(e)}")
                        current_month_data['Date'] = pd.Timestamp.now().date()
                    
                    for date in current_month_data['Date'].unique():
                        try:
                            date_data = current_month_data[current_month_data['Date'] == date]
                            date_mask = combined['Date'] == date
                            
                            if not date_mask.any():
                                logger.debug(f"Date {date} not found in combined data")
                                continue
                            
                            for col in ['Greek_Sentiment', 'Greek_Sentiment_Regime', 'SumVegaCall', 'SumVegaPut',
                                       'SumDeltaCall', 'SumDeltaPut', 'SumThetaCall', 'SumThetaPut']:
                                if col in date_data.columns and len(date_data[col]) > 0:
                                    weight = self.current_month_weight
                                    combined.loc[date_mask, col] += weight * date_data[col].values[0]
                        except Exception as e:
                            logger.warning(f"Error processing current month date {date}: {str(e)}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error combining expiry sentiment: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a valid but basic result
            if len(combined) == 0:
                combined = pd.DataFrame({
                    'Date': [pd.Timestamp.now().date()],
                    'Greek_Sentiment': [0.0],
                    'Greek_Sentiment_Regime': [0]
                })
        
        # Normalize Greek_Sentiment to ensure it's in [-1, 1] range
        if 'Greek_Sentiment' in combined.columns:
            max_val = combined['Greek_Sentiment'].abs().max()
            if max_val > 1:
                combined['Greek_Sentiment'] = combined['Greek_Sentiment'] / max_val
        
        # Handle extreme values in Greek_Sentiment_Regime
        if 'Greek_Sentiment_Regime' in combined.columns:
            combined['Greek_Sentiment_Regime'] = combined['Greek_Sentiment_Regime'].clip(-2, 2)
            
        return combined
    
    def _generate_synthetic_greek_data(self, data_frame, expiry_column):
        """
        Generate synthetic Greek data for testing when real data is not available.
        
        Args:
            data_frame (pd.DataFrame): Input data frame
            expiry_column (str): Name of the expiry column
            
        Returns:
            pd.DataFrame: Data frame with synthetic Greek data
        """
        logger.info("Generating synthetic Greek data for testing")
        
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Ensure Date and Time columns exist
        if 'Date' not in df.columns:
            df['Date'] = pd.Timestamp.now().date()
        if 'Time' not in df.columns:
            # Generate time sequence
            df['Time'] = pd.date_range(
                start='09:15:00', 
                periods=len(df), 
                freq='1min'
            ).strftime('%H:%M:%S')
        
        # Ensure expiry column exists
        if expiry_column not in df.columns:
            df[expiry_column] = np.random.choice(
                ['current_week', 'next_week', 'current_month'], 
                size=len(df),
                p=[0.6, 0.3, 0.1]
            )
        
        # Define strike levels (delta values)
        delta_values = np.round(np.arange(self.min_delta, self.max_delta + 0.1, 0.1), 1)
        
        # Generate time series data with mean-reversion for realistic movement
        def generate_time_series(n, mean=0, std=1, lag=0.9):
            # Generate more realistic time series with mean reversion
            series = np.zeros(n)
            series[0] = np.random.normal(mean, std)
            
            for i in range(1, n):
                # Mean reversion formula
                series[i] = lag * series[i-1] + (1-lag) * mean + np.random.normal(0, std)
            
            return series
        
        # Base price and volatility
        base_price = 100 + generate_time_series(len(df), mean=0, std=5, lag=0.95)
        vix = 20 + generate_time_series(len(df), mean=0, std=3, lag=0.9)
        
        # Generate columns with proper option Greeks pattern for each side, delta, and Greek type
        sides = {'Call': 'Call', 'Put': 'Put'}
        greek_types = ['Delta', 'Gamma', 'Theta', 'Vega']
        
        # Create base market trend that will influence Greeks
        market_trend = generate_time_series(len(df), mean=0, std=0.02, lag=0.8)
        
        # Create Greek data for each combination
        for side_name, side_value in sides.items():
            for delta in delta_values:
                for greek in greek_types:
                    column_name = f"{side_value}_{delta}_{greek}"
                    
                    # Base values depend on the Greek type
                    if greek == 'Delta':
                        # Delta values should be around the strike's delta value
                        base = delta if side_name == 'Call' else -delta
                        noise = 0.05
                        trend_factor = 0.1  # Market trend has less effect on delta
                    elif greek == 'Gamma':
                        # Gamma is highest ATM and decreases as we move away
                        base = 0.01 * (1 - abs(delta - 0.5) * 2)
                        noise = 0.002
                        trend_factor = 0.2
                    elif greek == 'Theta':
                        # Theta is negative and greatest ATM
                        base = -0.05 * (1 - abs(delta - 0.5))
                        noise = 0.01
                        trend_factor = 0.3
                    elif greek == 'Vega':
                        # Vega is highest ATM and decreases as we move away
                        base = 0.2 * (1 - abs(delta - 0.5) * 2)
                        noise = 0.05
                        trend_factor = 0.5  # Market trend has more effect on vega
                    
                    # Generate time series with the base value
                    time_series = base + generate_time_series(len(df), mean=0, std=noise, lag=0.7)
                    
                    # Add market trend influence (more when market moves against the option side)
                    trend_effect = np.zeros(len(df))
                    for i in range(len(df)):
                        # Call options gain value when market goes up, put options when market goes down
                        direction = 1 if side_name == 'Call' else -1
                        trend_effect[i] = direction * market_trend[i] * trend_factor
                    
                    # Add volatility effect (higher volatility increases vega and gamma)
                    vol_effect = np.zeros(len(df))
                    if greek in ['Vega', 'Gamma']:
                        for i in range(len(df)):
                            vol_factor = 0.01 if greek == 'Gamma' else 0.03
                            vol_effect[i] = (vix[i] - 20) * vol_factor
                    
                    # Combine effects
                    final_series = time_series + trend_effect + vol_effect
                    
                    # Store in dataframe
                    df[column_name] = final_series
        
        # Generate realistic sentiment based on the synthetic Greeks
        self._add_synthetic_sentiment(df)
        
        logger.info(f"Generated {len(delta_values) * len(sides) * len(greek_types)} synthetic Greek columns")
        return df
        
    def _add_synthetic_sentiment(self, df):
        """
        Add synthetic Greek sentiment based on the generated Greek data.
        
        Args:
            df (pd.DataFrame): Data frame with synthetic Greek data
        """
        # Calculate a realistic sentiment score based on the generated Greeks
        call_delta_columns = [col for col in df.columns if 'Call' in col and 'Delta' in col]
        put_delta_columns = [col for col in df.columns if 'Put' in col and 'Delta' in col]
        
        call_vega_columns = [col for col in df.columns if 'Call' in col and 'Vega' in col]
        put_vega_columns = [col for col in df.columns if 'Put' in col and 'Vega' in col]
        
        # Calculate net delta and vega
        if call_delta_columns and put_delta_columns:
            df['SumDeltaCall'] = df[call_delta_columns].sum(axis=1)
            df['SumDeltaPut'] = df[put_delta_columns].sum(axis=1)
            df['NetDelta'] = df['SumDeltaCall'] + df['SumDeltaPut']
        else:
            df['NetDelta'] = generate_time_series(len(df), mean=0, std=0.2, lag=0.8)
        
        if call_vega_columns and put_vega_columns:
            df['SumVegaCall'] = df[call_vega_columns].sum(axis=1)
            df['SumVegaPut'] = df[put_vega_columns].sum(axis=1)
            df['NetVega'] = df['SumVegaCall'] + df['SumVegaPut']
        else:
            df['NetVega'] = generate_time_series(len(df), mean=0, std=0.3, lag=0.7)
        
        # Calculate sentiment
        df['Greek_Sentiment'] = (
            self.delta_weight * df['NetDelta'] / df['NetDelta'].abs().max() +
            self.vega_weight * df['NetVega'] / df['NetVega'].abs().max()
        ) / (self.delta_weight + self.vega_weight)
        
        # Ensure values are in [-1, 1] range
        df['Greek_Sentiment'] = np.clip(df['Greek_Sentiment'], -1, 1)
        
        # Create sentiment regime (discretized version)
        df['Greek_Sentiment_Regime'] = pd.cut(
            df['Greek_Sentiment'], 
            bins=[-1.01, -0.6, -0.2, 0.2, 0.6, 1.01], 
            labels=[-2, -1, 0, 1, 2]
        ).astype(int)
        
        return df

    def generate_synthetic_data(self, data_frame, missing_greeks=None, missing_call_put=None, greek_columns=None, call_put_columns=None):
        """
        Legacy method for compatibility with older implementations.
        
        Args:
            data_frame (pd.DataFrame): Input data
            missing_greeks (list): List of missing Greek columns
            missing_call_put (list): List of missing Call/Put columns
            greek_columns (dict): Dictionary of Greek column names
            call_put_columns (dict): Dictionary of Call/Put column names
            
        Returns:
            pd.DataFrame: Data with synthetic columns added
        """
        logger.warning("Using legacy synthetic data generation method")
        return self._generate_synthetic_greek_data(data_frame, 'Expiry')
