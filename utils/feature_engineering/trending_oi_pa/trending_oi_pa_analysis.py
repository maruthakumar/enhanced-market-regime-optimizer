"""
Enhanced Trending OI with PA Analysis Module

This module implements comprehensive Trending Open Interest with Price Action analysis,
analyzing ATM plus 7 strikes above and 7 strikes below (total of 15 strikes)
and implementing advanced OI pattern recognition across these strikes.

Features:
- OI trends analysis across 15 strikes
- OI velocity and acceleration calculation
- Strike skew analysis for OI distribution
- OI divergence from price action detection
- Time decay impact on OI patterns
- Cross-expiry OI analysis
- Institutional vs. retail positioning analysis
- Historical pattern behavior analysis
- Short build-up, short covering, short unwinding, long build-up, and long unwinding detection
- Combined call-put pattern identification at the same strike prices
- Pattern divergence analysis
- Historical pattern performance tracking
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logger = logging.getLogger(__name__)

class TrendingOIWithPAAnalysis:
    """
    Enhanced Trending OI with PA Analysis.
    
    This class implements comprehensive analysis of Open Interest trends and their relationship
    with Price Action, focusing on ATM plus 7 strikes above and 7 strikes below
    (total of 15 strikes) and implementing advanced OI pattern recognition across these strikes.
    """
    
    def __init__(self, config=None):
        """
        Initialize Trending OI with PA Analysis.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        # Set default configuration values
        self.config = config or {}
        
        # Default weight in market regime classification
        self.default_weight = float(self.config.get('default_weight', 0.30))
        
        # Strike selection configuration
        self.strikes_above_atm = int(self.config.get('strikes_above_atm', 7))
        self.strikes_below_atm = int(self.config.get('strikes_below_atm', 7))
        
        # OI trend thresholds
        self.oi_increase_threshold = float(self.config.get('oi_increase_threshold', 0.05))  # 5% increase
        self.oi_decrease_threshold = float(self.config.get('oi_decrease_threshold', -0.05))  # 5% decrease
        
        # Price action thresholds
        self.price_increase_threshold = float(self.config.get('price_increase_threshold', 0.01))  # 1% increase
        self.price_decrease_threshold = float(self.config.get('price_decrease_threshold', -0.01))  # 1% decrease
        
        # Rolling window sizes
        self.short_window = int(self.config.get('short_window', 5))  # 5 periods
        self.medium_window = int(self.config.get('medium_window', 15))  # 15 periods
        self.long_window = int(self.config.get('long_window', 30))  # 30 periods
        
        # Trend strength thresholds
        self.strong_trend_threshold = float(self.config.get('strong_trend_threshold', 0.10))  # 10% change
        self.weak_trend_threshold = float(self.config.get('weak_trend_threshold', 0.05))  # 5% change
        
        # OI velocity and acceleration thresholds
        self.high_velocity_threshold = float(self.config.get('high_velocity_threshold', 0.03))  # 3% change per period
        self.high_acceleration_threshold = float(self.config.get('high_acceleration_threshold', 0.01))  # 1% change in velocity
        
        # Lot size thresholds for institutional vs. retail
        self.institutional_lot_size = int(self.config.get('institutional_lot_size', 100))  # 100 lots or more is institutional
        
        # Pattern recognition parameters
        self.pattern_lookback = int(self.config.get('pattern_lookback', 20))  # 20 periods lookback
        self.pattern_similarity_threshold = float(self.config.get('pattern_similarity_threshold', 0.80))  # 80% similarity
        
        # Dynamic weight adjustment parameters
        self.use_dynamic_weights = bool(self.config.get('use_dynamic_weights', True))
        self.learning_rate = float(self.config.get('learning_rate', 0.1))
        
        # Historical pattern analysis parameters
        self.history_window = int(self.config.get('history_window', 60))  # 60 periods for historical analysis
        self.pattern_performance_lookback = int(self.config.get('pattern_performance_lookback', 5))  # Look 5 periods ahead for performance
        self.pattern_history_file = self.config.get('pattern_history_file', 'pattern_history.pkl')
        self.min_pattern_occurrences = int(self.config.get('min_pattern_occurrences', 10))  # Minimum occurrences for reliable stats
        
        # Pattern divergence analysis parameters
        self.divergence_threshold = float(self.config.get('divergence_threshold', 0.3))  # 30% divergence threshold
        self.divergence_window = int(self.config.get('divergence_window', 10))  # 10 periods for divergence analysis
        
        # Initialize pattern history storage
        self.pattern_history = {
            'Strong_Bullish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
            'Mild_Bullish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
            'Neutral': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
            'Mild_Bearish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
            'Strong_Bearish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0}
        }
        
        # Load existing pattern history if available
        self._load_pattern_history()
        
        logger.info(f"Initialized Enhanced Trending OI with PA Analysis with default weight {self.default_weight}")
        logger.info(f"Using {self.strikes_above_atm} strikes above ATM and {self.strikes_below_atm} strikes below ATM")
        logger.info(f"Historical pattern analysis enabled with {self.history_window} periods window")
    
    def _load_pattern_history(self):
        """
        Load pattern history from file if available.
        """
        try:
            if os.path.exists(self.pattern_history_file):
                with open(self.pattern_history_file, 'rb') as f:
                    self.pattern_history = pickle.load(f)
                logger.info(f"Loaded pattern history from {self.pattern_history_file}")
            else:
                logger.info(f"No pattern history file found at {self.pattern_history_file}, using default values")
        except Exception as e:
            logger.error(f"Error loading pattern history: {str(e)}")
    
    def _save_pattern_history(self):
        """
        Save pattern history to file.
        """
        try:
            with open(self.pattern_history_file, 'wb') as f:
                pickle.dump(self.pattern_history, f)
            logger.info(f"Saved pattern history to {self.pattern_history_file}")
        except Exception as e:
            logger.error(f"Error saving pattern history: {str(e)}")
    
    def analyze_oi_patterns(self, data):
        """
        Analyze OI patterns from options data.
        
        Args:
            data (pd.DataFrame): Options data with OI
            
        Returns:
            pd.DataFrame: Data with OI pattern analysis
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['datetime', 'strike', 'option_type', 'open_interest', 'price', 'underlying_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to find alternative column names
            column_mapping = {
                'datetime': ['timestamp', 'date_time', 'time'],
                'strike': ['Strike', 'strike_price', 'STRIKE'],
                'option_type': ['type', 'call_put', 'cp', 'option_type'],
                'open_interest': ['OI', 'OPEN_INTEREST', 'oi'],
                'price': ['close', 'Close', 'CLOSE', 'last_price'],
                'underlying_price': ['underlying', 'Underlying', 'spot_price', 'index_price']
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
        
        # Sort by datetime and strike
        df = df.sort_values(['datetime', 'strike'])
        
        # Get unique timestamps
        timestamps = df['datetime'].unique()
        
        # Process each timestamp
        result_dfs = []
        
        for i, timestamp in enumerate(timestamps):
            # Get data for this timestamp
            timestamp_data = df[df['datetime'] == timestamp].copy()
            
            # Identify ATM strike
            atm_strike = self._identify_atm_strike(timestamp_data)
            
            # Select strikes to analyze
            selected_strikes = self._select_strikes(timestamp_data, atm_strike)
            
            # Filter data for selected strikes
            filtered_data = timestamp_data[timestamp_data['strike'].isin(selected_strikes)].copy()
            
            # Skip if not enough data
            if len(filtered_data) < 2:
                result_dfs.append(filtered_data)
                continue
            
            # Calculate OI patterns
            filtered_data = self._calculate_oi_patterns(filtered_data, i, timestamps)
            
            # Append to results
            result_dfs.append(filtered_data)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=False)
        
        # Sort by datetime and strike
        result = result.sort_values(['datetime', 'strike'])
        
        # Analyze historical pattern behavior if we have enough data
        if len(timestamps) > self.history_window:
            result = self._analyze_historical_pattern_behavior(result)
        
        # Analyze pattern divergence
        result = self._analyze_pattern_divergence(result)
        
        return result
    
    def _identify_atm_strike(self, data):
        """
        Identify ATM strike.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp
            
        Returns:
            float: ATM strike price
        """
        # Get underlying price
        if 'underlying_price' in data.columns:
            underlying_price = data['underlying_price'].iloc[0]
        else:
            logger.warning("No underlying_price column, using mean of strikes")
            underlying_price = data['strike'].mean()
        
        # Get unique strikes
        unique_strikes = data['strike'].unique()
        
        # Sort strikes
        unique_strikes = np.sort(unique_strikes)
        
        # Find closest strike
        atm_strike = unique_strikes[np.abs(unique_strikes - underlying_price).argmin()]
        
        return atm_strike
    
    def _select_strikes(self, data, atm_strike):
        """
        Select strikes to analyze.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp
            atm_strike (float): ATM strike price
            
        Returns:
            list: List of selected strikes
        """
        # Get unique strikes
        unique_strikes = np.sort(data['strike'].unique())
        
        # Find index of ATM strike
        atm_index = np.where(unique_strikes == atm_strike)[0][0]
        
        # Select strikes above ATM
        strikes_above = unique_strikes[atm_index+1:atm_index+1+self.strikes_above_atm] if atm_index+1+self.strikes_above_atm <= len(unique_strikes) else unique_strikes[atm_index+1:]
        
        # Select strikes below ATM
        strikes_below = unique_strikes[max(0, atm_index-self.strikes_below_atm):atm_index] if atm_index-self.strikes_below_atm >= 0 else unique_strikes[:atm_index]
        
        # Combine strikes
        selected_strikes = list(strikes_below) + [atm_strike] + list(strikes_above)
        
        return selected_strikes
    
    def _calculate_oi_patterns(self, data, timestamp_index, all_timestamps):
        """
        Calculate OI patterns for a single timestamp.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            timestamp_index (int): Index of current timestamp
            all_timestamps (array): Array of all timestamps
            
        Returns:
            pd.DataFrame: Data with OI patterns
        """
        # Make a copy
        df = data.copy()
        
        # Calculate basic OI metrics
        df = self._calculate_basic_oi_metrics(df)
        
        # Calculate OI velocity and acceleration if we have previous timestamps
        if timestamp_index > 0:
            df = self._calculate_oi_velocity_acceleration(df, timestamp_index, all_timestamps)
        
        # Calculate strike skew
        df = self._calculate_strike_skew(df)
        
        # Calculate OI divergence from price action
        df = self._calculate_oi_price_divergence(df)
        
        # Calculate time decay impact if we have expiry information
        if 'expiry' in df.columns or 'dte' in df.columns:
            df = self._calculate_time_decay_impact(df)
        
        # Calculate institutional vs. retail positioning
        df = self._calculate_institutional_retail(df)
        
        # Identify OI patterns
        df = self._identify_oi_patterns(df)
        
        # Calculate combined call-put patterns
        df = self._calculate_combined_patterns(df)
        
        return df
    
    def _calculate_basic_oi_metrics(self, data):
        """
        Calculate basic OI metrics.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with basic OI metrics
        """
        # Make a copy
        df = data.copy()
        
        # Calculate call and put OI sums
        call_oi_sum = df[df['option_type'] == 'call']['open_interest'].sum()
        put_oi_sum = df[df['option_type'] == 'put']['open_interest'].sum()
        
        # Calculate put-call ratio
        put_call_ratio = put_oi_sum / call_oi_sum if call_oi_sum > 0 else np.nan
        
        # Add to dataframe
        df['call_oi_sum'] = call_oi_sum
        df['put_oi_sum'] = put_oi_sum
        df['put_call_ratio'] = put_call_ratio
        
        # Calculate OI by strike
        for strike in df['strike'].unique():
            strike_data = df[df['strike'] == strike]
            call_oi = strike_data[strike_data['option_type'] == 'call']['open_interest'].sum()
            put_oi = strike_data[strike_data['option_type'] == 'put']['open_interest'].sum()
            strike_put_call_ratio = put_oi / call_oi if call_oi > 0 else np.nan
            
            df.loc[df['strike'] == strike, 'strike_call_oi'] = call_oi
            df.loc[df['strike'] == strike, 'strike_put_oi'] = put_oi
            df.loc[df['strike'] == strike, 'strike_put_call_ratio'] = strike_put_call_ratio
        
        return df
    
    def _calculate_oi_velocity_acceleration(self, data, timestamp_index, all_timestamps):
        """
        Calculate OI velocity and acceleration.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            timestamp_index (int): Index of current timestamp
            all_timestamps (array): Array of all timestamps
            
        Returns:
            pd.DataFrame: Data with OI velocity and acceleration
        """
        # Make a copy
        df = data.copy()
        
        # Get current timestamp
        current_timestamp = all_timestamps[timestamp_index]
        
        # Get previous timestamp
        prev_timestamp = all_timestamps[timestamp_index - 1]
        
        # Get data for previous timestamp
        prev_data = df.copy()
        prev_data['datetime'] = prev_timestamp
        
        # Calculate velocity (rate of change)
        for strike in df['strike'].unique():
            strike_data = df[df['strike'] == strike]
            
            # Call OI velocity
            call_oi = strike_data[strike_data['option_type'] == 'call']['open_interest'].sum()
            prev_call_oi = prev_data[prev_data['strike'] == strike]
            prev_call_oi = prev_call_oi[prev_call_oi['option_type'] == 'call']['open_interest'].sum()
            call_oi_velocity = (call_oi - prev_call_oi) / prev_call_oi if prev_call_oi > 0 else 0
            
            # Put OI velocity
            put_oi = strike_data[strike_data['option_type'] == 'put']['open_interest'].sum()
            prev_put_oi = prev_data[prev_data['strike'] == strike]
            prev_put_oi = prev_put_oi[prev_put_oi['option_type'] == 'put']['open_interest'].sum()
            put_oi_velocity = (put_oi - prev_put_oi) / prev_put_oi if prev_put_oi > 0 else 0
            
            # Update dataframe
            df.loc[df['strike'] == strike, 'call_oi_velocity'] = call_oi_velocity
            df.loc[df['strike'] == strike, 'put_oi_velocity'] = put_oi_velocity
        
        # Calculate acceleration (rate of change of velocity) if we have at least 2 previous timestamps
        if timestamp_index > 1:
            # Get previous previous timestamp
            prev_prev_timestamp = all_timestamps[timestamp_index - 2]
            
            # Get data for previous previous timestamp
            prev_prev_data = df.copy()
            prev_prev_data['datetime'] = prev_prev_timestamp
            
            for strike in df['strike'].unique():
                strike_data = df[df['strike'] == strike]
                
                # Call OI acceleration
                call_oi_velocity = strike_data[strike_data['option_type'] == 'call']['call_oi_velocity'].iloc[0] if len(strike_data) > 0 else 0
                prev_call_oi = prev_data[prev_data['strike'] == strike]
                prev_call_oi_velocity = prev_call_oi[prev_call_oi['option_type'] == 'call']['call_oi_velocity'].iloc[0] if len(prev_call_oi) > 0 else 0
                call_oi_acceleration = call_oi_velocity - prev_call_oi_velocity
                
                # Put OI acceleration
                put_oi_velocity = strike_data[strike_data['option_type'] == 'put']['put_oi_velocity'].iloc[0] if len(strike_data) > 0 else 0
                prev_put_oi = prev_data[prev_data['strike'] == strike]
                prev_put_oi_velocity = prev_put_oi[prev_put_oi['option_type'] == 'put']['put_oi_velocity'].iloc[0] if len(prev_put_oi) > 0 else 0
                put_oi_acceleration = put_oi_velocity - prev_put_oi_velocity
                
                # Update dataframe
                df.loc[df['strike'] == strike, 'call_oi_acceleration'] = call_oi_acceleration
                df.loc[df['strike'] == strike, 'put_oi_acceleration'] = put_oi_acceleration
        
        # Calculate price velocity
        if 'underlying_price' in df.columns:
            current_price = df['underlying_price'].iloc[0]
            prev_price = prev_data['underlying_price'].iloc[0]
            price_velocity = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            df['price_velocity'] = price_velocity
        
        return df
    
    def _calculate_strike_skew(self, data):
        """
        Calculate strike skew.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with strike skew
        """
        # Make a copy
        df = data.copy()
        
        # Calculate call OI skew
        call_oi_by_strike = df[df['option_type'] == 'call'].groupby('strike')['open_interest'].sum()
        if len(call_oi_by_strike) > 2:
            call_oi_skew = skew(call_oi_by_strike.values)
            call_oi_kurtosis = kurtosis(call_oi_by_strike.values)
            df['call_oi_skew'] = call_oi_skew
            df['call_oi_kurtosis'] = call_oi_kurtosis
        
        # Calculate put OI skew
        put_oi_by_strike = df[df['option_type'] == 'put'].groupby('strike')['open_interest'].sum()
        if len(put_oi_by_strike) > 2:
            put_oi_skew = skew(put_oi_by_strike.values)
            put_oi_kurtosis = kurtosis(put_oi_by_strike.values)
            df['put_oi_skew'] = put_oi_skew
            df['put_oi_kurtosis'] = put_oi_kurtosis
        
        return df
    
    def _calculate_oi_price_divergence(self, data):
        """
        Calculate OI divergence from price action.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with OI price divergence
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have price velocity and OI velocity
        if 'price_velocity' in df.columns and 'call_oi_velocity' in df.columns and 'put_oi_velocity' in df.columns:
            # Get price velocity
            price_velocity = df['price_velocity'].iloc[0]
            
            # Calculate call OI divergence
            for strike in df['strike'].unique():
                strike_data = df[df['strike'] == strike]
                
                # Call OI divergence
                call_data = strike_data[strike_data['option_type'] == 'call']
                if len(call_data) > 0:
                    call_oi_velocity = call_data['call_oi_velocity'].iloc[0]
                    call_oi_divergence = 1 if (price_velocity > 0 and call_oi_velocity < 0) or (price_velocity < 0 and call_oi_velocity > 0) else 0
                    df.loc[(df['strike'] == strike) & (df['option_type'] == 'call'), 'call_oi_divergence'] = call_oi_divergence
                
                # Put OI divergence
                put_data = strike_data[strike_data['option_type'] == 'put']
                if len(put_data) > 0:
                    put_oi_velocity = put_data['put_oi_velocity'].iloc[0]
                    put_oi_divergence = 1 if (price_velocity > 0 and put_oi_velocity > 0) or (price_velocity < 0 and put_oi_velocity < 0) else 0
                    df.loc[(df['strike'] == strike) & (df['option_type'] == 'put'), 'put_oi_divergence'] = put_oi_divergence
        
        return df
    
    def _calculate_time_decay_impact(self, data):
        """
        Calculate time decay impact on OI patterns.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with time decay impact
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have expiry or dte information
        if 'expiry' in df.columns:
            # Calculate days to expiration
            if not pd.api.types.is_datetime64_any_dtype(df['expiry']):
                try:
                    df['expiry'] = pd.to_datetime(df['expiry'])
                except:
                    logger.warning("Failed to convert expiry column to datetime format")
                    return df
            
            # Calculate DTE
            df['dte'] = (df['expiry'] - df['datetime']).dt.days
        
        # Adjust OI velocity based on DTE if we have DTE information
        if 'dte' in df.columns and 'call_oi_velocity' in df.columns and 'put_oi_velocity' in df.columns:
            # Calculate DTE factor (higher impact for shorter DTE)
            df['dte_factor'] = 1 / (df['dte'] + 1)
            
            # Adjust OI velocity
            df['call_oi_velocity_adjusted'] = df['call_oi_velocity'] * (1 - df['dte_factor'])
            df['put_oi_velocity_adjusted'] = df['put_oi_velocity'] * (1 - df['dte_factor'])
        
        return df
    
    def _calculate_institutional_retail(self, data):
        """
        Calculate institutional vs. retail positioning.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with institutional vs. retail positioning
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have lot size information
        if 'lot_size' in df.columns:
            # Identify institutional trades
            df['is_institutional'] = df['lot_size'] >= self.institutional_lot_size
            
            # Calculate institutional OI
            for strike in df['strike'].unique():
                strike_data = df[df['strike'] == strike]
                
                # Call institutional OI
                call_data = strike_data[(strike_data['option_type'] == 'call') & (strike_data['is_institutional'])]
                call_institutional_oi = call_data['open_interest'].sum() if len(call_data) > 0 else 0
                call_retail_oi = strike_data[(strike_data['option_type'] == 'call') & (~strike_data['is_institutional'])]['open_interest'].sum() if len(strike_data) > 0 else 0
                
                # Put institutional OI
                put_data = strike_data[(strike_data['option_type'] == 'put') & (strike_data['is_institutional'])]
                put_institutional_oi = put_data['open_interest'].sum() if len(put_data) > 0 else 0
                put_retail_oi = strike_data[(strike_data['option_type'] == 'put') & (~strike_data['is_institutional'])]['open_interest'].sum() if len(strike_data) > 0 else 0
                
                # Update dataframe
                df.loc[df['strike'] == strike, 'call_institutional_oi'] = call_institutional_oi
                df.loc[df['strike'] == strike, 'call_retail_oi'] = call_retail_oi
                df.loc[df['strike'] == strike, 'put_institutional_oi'] = put_institutional_oi
                df.loc[df['strike'] == strike, 'put_retail_oi'] = put_retail_oi
                
                # Calculate institutional ratios
                df.loc[df['strike'] == strike, 'call_institutional_ratio'] = call_institutional_oi / (call_institutional_oi + call_retail_oi) if (call_institutional_oi + call_retail_oi) > 0 else 0
                df.loc[df['strike'] == strike, 'put_institutional_ratio'] = put_institutional_oi / (put_institutional_oi + put_retail_oi) if (put_institutional_oi + put_retail_oi) > 0 else 0
        
        return df
    
    def _identify_oi_patterns(self, data):
        """
        Identify OI patterns.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with OI patterns
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have OI velocity and price velocity
        if 'call_oi_velocity' in df.columns and 'put_oi_velocity' in df.columns and 'price_velocity' in df.columns:
            # Initialize pattern columns
            df['call_oi_pattern'] = 'Unknown'
            df['put_oi_pattern'] = 'Unknown'
            
            # Process each strike
            for strike in df['strike'].unique():
                strike_data = df[df['strike'] == strike]
                
                # Get call data
                call_data = strike_data[strike_data['option_type'] == 'call']
                if len(call_data) > 0:
                    call_oi_velocity = call_data['call_oi_velocity'].iloc[0]
                    price_velocity = call_data['price_velocity'].iloc[0]
                    
                    # Identify call OI pattern
                    if call_oi_velocity > 0 and price_velocity > 0:
                        pattern = 'Long_Build_Up'  # OI up, price up
                    elif call_oi_velocity < 0 and price_velocity < 0:
                        pattern = 'Long_Unwinding'  # OI down, price down
                    elif call_oi_velocity > 0 and price_velocity < 0:
                        pattern = 'Short_Build_Up'  # OI up, price down
                    elif call_oi_velocity < 0 and price_velocity > 0:
                        pattern = 'Short_Covering'  # OI down, price up
                    else:
                        pattern = 'Neutral'
                    
                    df.loc[(df['strike'] == strike) & (df['option_type'] == 'call'), 'call_oi_pattern'] = pattern
                
                # Get put data
                put_data = strike_data[strike_data['option_type'] == 'put']
                if len(put_data) > 0:
                    put_oi_velocity = put_data['put_oi_velocity'].iloc[0]
                    price_velocity = put_data['price_velocity'].iloc[0]
                    
                    # Identify put OI pattern
                    if put_oi_velocity > 0 and price_velocity < 0:
                        pattern = 'Long_Build_Up'  # OI up, price down (for puts)
                    elif put_oi_velocity < 0 and price_velocity > 0:
                        pattern = 'Long_Unwinding'  # OI down, price up (for puts)
                    elif put_oi_velocity > 0 and price_velocity > 0:
                        pattern = 'Short_Build_Up'  # OI up, price up (for puts)
                    elif put_oi_velocity < 0 and price_velocity < 0:
                        pattern = 'Short_Covering'  # OI down, price down (for puts)
                    else:
                        pattern = 'Neutral'
                    
                    df.loc[(df['strike'] == strike) & (df['option_type'] == 'put'), 'put_oi_pattern'] = pattern
        
        return df
    
    def _calculate_combined_patterns(self, data):
        """
        Calculate combined call-put patterns.
        
        Args:
            data (pd.DataFrame): Data for a single timestamp and selected strikes
            
        Returns:
            pd.DataFrame: Data with combined patterns
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have call and put OI patterns
        if 'call_oi_pattern' in df.columns and 'put_oi_pattern' in df.columns:
            # Initialize combined pattern column
            df['combined_oi_pattern'] = 'Unknown'
            
            # Process each strike
            for strike in df['strike'].unique():
                # Get call pattern
                call_pattern = df[(df['strike'] == strike) & (df['option_type'] == 'call')]['call_oi_pattern'].iloc[0] if len(df[(df['strike'] == strike) & (df['option_type'] == 'call')]) > 0 else 'Unknown'
                
                # Get put pattern
                put_pattern = df[(df['strike'] == strike) & (df['option_type'] == 'put')]['put_oi_pattern'].iloc[0] if len(df[(df['strike'] == strike) & (df['option_type'] == 'put')]) > 0 else 'Unknown'
                
                # Calculate combined pattern
                combined_pattern = 'Neutral'
                
                # Strong bullish patterns - Corrected from option seller's perspective
                if (call_pattern == 'Long_Build_Up' and (put_pattern == 'Short_Build_Up' or put_pattern == 'Long_Unwinding')) or \
                   (call_pattern == 'Short_Covering' and (put_pattern == 'Short_Build_Up' or put_pattern == 'Long_Unwinding')):
                    combined_pattern = 'Strong_Bullish'
                
                # Mild bullish patterns
                elif (call_pattern == 'Long_Build_Up') or \
                     (call_pattern == 'Short_Covering') or \
                     (put_pattern == 'Long_Unwinding') or \
                     (put_pattern == 'Short_Covering'):  # Put Short_Covering is bullish
                    combined_pattern = 'Mild_Bullish'
                
                # Strong bearish patterns - Corrected from option seller's perspective
                elif (put_pattern == 'Long_Build_Up' and (call_pattern == 'Short_Build_Up' or call_pattern == 'Long_Unwinding')) or \
                     (put_pattern == 'Short_Covering' and (call_pattern == 'Short_Build_Up' or call_pattern == 'Long_Unwinding')):
                    combined_pattern = 'Strong_Bearish'
                
                # Mild bearish patterns
                elif (put_pattern == 'Long_Build_Up') or \
                     (call_pattern == 'Long_Unwinding') or \
                     (call_pattern == 'Short_Build_Up'):
                    combined_pattern = 'Mild_Bearish'
                
                # Update dataframe
                df.loc[df['strike'] == strike, 'combined_oi_pattern'] = combined_pattern
            
            # Calculate overall pattern
            pattern_counts = df['combined_oi_pattern'].value_counts()
            
            if 'Strong_Bullish' in pattern_counts and pattern_counts['Strong_Bullish'] >= 3:
                df['overall_oi_pattern'] = 'Strong_Bullish'
            elif 'Strong_Bearish' in pattern_counts and pattern_counts['Strong_Bearish'] >= 3:
                df['overall_oi_pattern'] = 'Strong_Bearish'
            elif 'Mild_Bullish' in pattern_counts and pattern_counts['Mild_Bullish'] >= 3:
                df['overall_oi_pattern'] = 'Mild_Bullish'
            elif 'Mild_Bearish' in pattern_counts and pattern_counts['Mild_Bearish'] >= 3:
                df['overall_oi_pattern'] = 'Mild_Bearish'
            else:
                df['overall_oi_pattern'] = 'Neutral'
        
        return df
    
    def _analyze_historical_pattern_behavior(self, data):
        """
        Analyze historical pattern behavior.
        
        Args:
            data (pd.DataFrame): Data with OI patterns
            
        Returns:
            pd.DataFrame: Data with historical pattern behavior analysis
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have overall OI pattern and underlying price
        if 'overall_oi_pattern' in df.columns and 'underlying_price' in df.columns:
            # Get unique timestamps
            timestamps = df['datetime'].unique()
            timestamps = sorted(timestamps)
            
            # Skip if not enough timestamps
            if len(timestamps) <= self.pattern_performance_lookback:
                logger.warning(f"Not enough timestamps for historical pattern analysis: {len(timestamps)} <= {self.pattern_performance_lookback}")
                return df
            
            # Initialize pattern performance tracking
            pattern_performance = {}
            
            # Process each timestamp except the last few (need future data for performance)
            for i in range(len(timestamps) - self.pattern_performance_lookback):
                # Get current timestamp
                current_timestamp = timestamps[i]
                
                # Get current data
                current_data = df[df['datetime'] == current_timestamp]
                
                # Get current pattern
                current_pattern = current_data['overall_oi_pattern'].iloc[0] if len(current_data) > 0 else 'Unknown'
                
                # Skip if unknown pattern
                if current_pattern == 'Unknown':
                    continue
                
                # Get current price
                current_price = current_data['underlying_price'].iloc[0]
                
                # Get future timestamp
                future_timestamp = timestamps[i + self.pattern_performance_lookback]
                
                # Get future data
                future_data = df[df['datetime'] == future_timestamp]
                
                # Get future price
                future_price = future_data['underlying_price'].iloc[0] if len(future_data) > 0 else current_price
                
                # Calculate return
                pattern_return = (future_price - current_price) / current_price
                
                # Update pattern performance
                if current_pattern not in pattern_performance:
                    pattern_performance[current_pattern] = {'returns': [], 'timestamps': []}
                
                pattern_performance[current_pattern]['returns'].append(pattern_return)
                pattern_performance[current_pattern]['timestamps'].append(current_timestamp)
            
            # Calculate pattern statistics
            for pattern, data in pattern_performance.items():
                returns = data['returns']
                
                if len(returns) >= self.min_pattern_occurrences:
                    # Calculate statistics
                    avg_return = np.mean(returns)
                    success_rate = np.mean([1 if (pattern.endswith('Bullish') and r > 0) or (pattern.endswith('Bearish') and r < 0) else 0 for r in returns])
                    
                    # Update pattern history
                    if pattern in self.pattern_history:
                        self.pattern_history[pattern]['occurrences'] += len(returns)
                        self.pattern_history[pattern]['success_rate'] = (self.pattern_history[pattern]['success_rate'] * (self.pattern_history[pattern]['occurrences'] - len(returns)) + success_rate * len(returns)) / self.pattern_history[pattern]['occurrences']
                        self.pattern_history[pattern]['avg_return'] = (self.pattern_history[pattern]['avg_return'] * (self.pattern_history[pattern]['occurrences'] - len(returns)) + avg_return * len(returns)) / self.pattern_history[pattern]['occurrences']
                    else:
                        self.pattern_history[pattern] = {
                            'occurrences': len(returns),
                            'success_rate': success_rate,
                            'avg_return': avg_return,
                            'avg_duration': self.pattern_performance_lookback
                        }
                    
                    # Add historical performance to dataframe
                    for timestamp in data['timestamps']:
                        df.loc[df['datetime'] == timestamp, f'{pattern}_historical_success_rate'] = success_rate
                        df.loc[df['datetime'] == timestamp, f'{pattern}_historical_avg_return'] = avg_return
            
            # Save updated pattern history
            self._save_pattern_history()
            
            # Add pattern confidence based on historical performance
            for timestamp in timestamps:
                timestamp_data = df[df['datetime'] == timestamp]
                pattern = timestamp_data['overall_oi_pattern'].iloc[0] if len(timestamp_data) > 0 else 'Unknown'
                
                if pattern in self.pattern_history and self.pattern_history[pattern]['occurrences'] >= self.min_pattern_occurrences:
                    success_rate = self.pattern_history[pattern]['success_rate']
                    df.loc[df['datetime'] == timestamp, 'pattern_confidence'] = success_rate
                else:
                    df.loc[df['datetime'] == timestamp, 'pattern_confidence'] = 0.5  # Neutral confidence if not enough history
        
        return df
    
    def _analyze_pattern_divergence(self, data):
        """
        Analyze pattern divergence between components.
        
        Args:
            data (pd.DataFrame): Data with OI patterns
            
        Returns:
            pd.DataFrame: Data with pattern divergence analysis
        """
        # Make a copy
        df = data.copy()
        
        # Check if we have overall OI pattern
        if 'overall_oi_pattern' in df.columns:
            # Get unique timestamps
            timestamps = df['datetime'].unique()
            timestamps = sorted(timestamps)
            
            # Skip if not enough timestamps
            if len(timestamps) <= self.divergence_window:
                logger.warning(f"Not enough timestamps for divergence analysis: {len(timestamps)} <= {self.divergence_window}")
                return df
            
            # Initialize divergence tracking
            divergence_tracking = {}
            
            # Process each timestamp
            for i in range(len(timestamps)):
                # Get current timestamp
                current_timestamp = timestamps[i]
                
                # Get current data
                current_data = df[df['datetime'] == current_timestamp]
                
                # Get current pattern
                current_pattern = current_data['overall_oi_pattern'].iloc[0] if len(current_data) > 0 else 'Unknown'
                
                # Skip if unknown pattern
                if current_pattern == 'Unknown':
                    continue
                
                # Check for divergence with other components
                if 'call_oi_skew' in current_data.columns and 'put_oi_skew' in current_data.columns:
                    call_skew = current_data['call_oi_skew'].iloc[0]
                    put_skew = current_data['put_oi_skew'].iloc[0]
                    
                    # Detect divergence between call and put skew
                    skew_divergence = False
                    if (current_pattern.endswith('Bullish') and call_skew < 0 and put_skew > 0) or \
                       (current_pattern.endswith('Bearish') and call_skew > 0 and put_skew < 0):
                        skew_divergence = True
                    
                    df.loc[df['datetime'] == current_timestamp, 'skew_divergence'] = skew_divergence
                
                # Check for divergence with institutional positioning
                if 'call_institutional_ratio' in current_data.columns and 'put_institutional_ratio' in current_data.columns:
                    call_inst_ratio = current_data['call_institutional_ratio'].iloc[0]
                    put_inst_ratio = current_data['put_institutional_ratio'].iloc[0]
                    
                    # Detect divergence between retail and institutional positioning
                    inst_divergence = False
                    if (current_pattern.endswith('Bullish') and call_inst_ratio < 0.5 and put_inst_ratio > 0.5) or \
                       (current_pattern.endswith('Bearish') and call_inst_ratio > 0.5 and put_inst_ratio < 0.5):
                        inst_divergence = True
                    
                    df.loc[df['datetime'] == current_timestamp, 'institutional_divergence'] = inst_divergence
                
                # Calculate overall divergence score
                divergence_score = 0
                divergence_count = 0
                
                if 'skew_divergence' in df.columns:
                    divergence_score += df.loc[df['datetime'] == current_timestamp, 'skew_divergence'].iloc[0]
                    divergence_count += 1
                
                if 'institutional_divergence' in df.columns:
                    divergence_score += df.loc[df['datetime'] == current_timestamp, 'institutional_divergence'].iloc[0]
                    divergence_count += 1
                
                if divergence_count > 0:
                    df.loc[df['datetime'] == current_timestamp, 'divergence_score'] = divergence_score / divergence_count
                else:
                    df.loc[df['datetime'] == current_timestamp, 'divergence_score'] = 0
                
                # Adjust pattern confidence based on divergence
                if 'pattern_confidence' in df.columns:
                    pattern_confidence = df.loc[df['datetime'] == current_timestamp, 'pattern_confidence'].iloc[0]
                    divergence_score = df.loc[df['datetime'] == current_timestamp, 'divergence_score'].iloc[0]
                    
                    # Reduce confidence if high divergence
                    if divergence_score > self.divergence_threshold:
                        adjusted_confidence = pattern_confidence * (1 - divergence_score)
                        df.loc[df['datetime'] == current_timestamp, 'adjusted_pattern_confidence'] = adjusted_confidence
                    else:
                        df.loc[df['datetime'] == current_timestamp, 'adjusted_pattern_confidence'] = pattern_confidence
        
        return df
    
    def visualize_oi_patterns(self, data, output_dir):
        """
        Visualize OI patterns.
        
        Args:
            data (pd.DataFrame): Data with OI patterns
            output_dir (str): Output directory for visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if we have the required columns
            if 'strike' in data.columns and 'call_oi_pattern' in data.columns and 'put_oi_pattern' in data.columns:
                # Get unique timestamps
                timestamps = data['datetime'].unique()
                
                # Process each timestamp
                for timestamp in timestamps:
                    # Get data for this timestamp
                    timestamp_data = data[data['datetime'] == timestamp].copy()
                    
                    # Create a figure with subplots
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Plot call OI patterns
                    call_data = timestamp_data[timestamp_data['option_type'] == 'call'].sort_values('strike')
                    sns.barplot(x='strike', y='open_interest', hue='call_oi_pattern', data=call_data, ax=axes[0])
                    axes[0].set_title(f'Call OI Patterns at {timestamp}')
                    axes[0].set_xlabel('Strike')
                    axes[0].set_ylabel('Open Interest')
                    axes[0].tick_params(axis='x', rotation=45)
                    
                    # Plot put OI patterns
                    put_data = timestamp_data[timestamp_data['option_type'] == 'put'].sort_values('strike')
                    sns.barplot(x='strike', y='open_interest', hue='put_oi_pattern', data=put_data, ax=axes[1])
                    axes[1].set_title(f'Put OI Patterns at {timestamp}')
                    axes[1].set_xlabel('Strike')
                    axes[1].set_ylabel('Open Interest')
                    axes[1].tick_params(axis='x', rotation=45)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save figure
                    timestamp_str = str(timestamp).replace(':', '-').replace(' ', '_')
                    plt.savefig(os.path.join(output_dir, f'oi_patterns_{timestamp_str}.png'))
                    plt.close()
                
                # Plot overall patterns over time
                if 'overall_oi_pattern' in data.columns:
                    plt.figure(figsize=(12, 6))
                    pattern_counts = data.groupby(['datetime', 'overall_oi_pattern']).size().unstack().fillna(0)
                    pattern_counts.plot(kind='bar', stacked=True)
                    plt.title('Overall OI Patterns Over Time')
                    plt.xlabel('Timestamp')
                    plt.ylabel('Count')
                    plt.legend(title='Pattern')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'overall_oi_patterns.png'))
                    plt.close()
                
                # Plot historical pattern performance if available
                if 'pattern_confidence' in data.columns:
                    plt.figure(figsize=(12, 6))
                    data.groupby('datetime')['pattern_confidence'].mean().plot()
                    plt.title('Pattern Confidence Over Time')
                    plt.xlabel('Timestamp')
                    plt.ylabel('Confidence')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'pattern_confidence.png'))
                    plt.close()
                
                # Plot divergence score if available
                if 'divergence_score' in data.columns:
                    plt.figure(figsize=(12, 6))
                    data.groupby('datetime')['divergence_score'].mean().plot()
                    plt.title('Component Divergence Score Over Time')
                    plt.xlabel('Timestamp')
                    plt.ylabel('Divergence Score')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'divergence_score.png'))
                    plt.close()
            
            logger.info(f"Saved OI pattern visualizations to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error visualizing OI patterns: {str(e)}")
    
    def get_pattern_history_stats(self):
        """
        Get pattern history statistics.
        
        Returns:
            dict: Pattern history statistics
        """
        return self.pattern_history
    
    def get_weight(self):
        """
        Get weight for market regime classification.
        
        Returns:
            float: Weight
        """
        return self.default_weight
