"""
Consolidated Greek Sentiment Analysis Module

This module implements Greek sentiment analysis for options trading,
focusing on tracking Vega, Delta, and Theta changes across strikes in real-time
to build a dynamic sentiment gauge.

Features:
- Real-time tracking of Greeks changes
- Strike/expiry universe definition (delta range 0.5 to 0.01)
- Near expiry (70% weight) and next expiry (30% weight) calculations
- Minute-to-minute changes calculation
- Aggregation of changes for calls vs. puts
- Threshold-based sentiment classification
- DTE-specific calculations
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger(__name__)

class GreekSentimentAnalysis:
    """
    Consolidated Greek Sentiment Analysis.
    
    This class implements Greek sentiment analysis for options trading,
    focusing on tracking Vega, Delta, and Theta changes across strikes in real-time
    to build a dynamic sentiment gauge.
    
    Implementation follows specifications in Greek_sentiment.md with delta range
    from 0.5 to 0.01 and expiry weightage: 70% for near expiry and 30% for next expiry.
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
        
        # Delta range for analysis
        self.delta_range = {
            'call': {
                'min': float(self.config.get('call_delta_min', 0.01)),
                'max': float(self.config.get('call_delta_max', 0.50))
            },
            'put': {
                'min': float(self.config.get('put_delta_min', -0.50)),
                'max': float(self.config.get('put_delta_max', -0.01))
            }
        }
        
        # Default weight in market regime classification (40%)
        self.default_weight = float(self.config.get('default_weight', 0.40))
        
        # Greek component weights
        self.greek_weights = {
            'vega': float(self.config.get('vega_weight', 0.40)),
            'delta': float(self.config.get('delta_weight', 0.40)),
            'theta': float(self.config.get('theta_weight', 0.20))
        }
        
        # Strike category weights
        self.strike_weights = {
            'atm': float(self.config.get('atm_strike_weight', 0.40)),
            'near_otm': float(self.config.get('near_otm_strike_weight', 0.30)),
            'mid_otm': float(self.config.get('mid_otm_strike_weight', 0.20)),
            'far_otm': float(self.config.get('far_otm_strike_weight', 0.10))
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'strong_bullish': float(self.config.get('strong_bullish_threshold', 0.5)),
            'mild_bullish': float(self.config.get('mild_bullish_threshold', 0.1)),
            'mild_bearish': float(self.config.get('mild_bearish_threshold', -0.1)),
            'strong_bearish': float(self.config.get('strong_bearish_threshold', -0.5))
        }
        
        logger.info(f"Initialized Greek Sentiment Analysis with default weight {self.default_weight}")
        logger.info(f"Using expiry weights: near={self.expiry_weights['near']}, next={self.expiry_weights['next']}")
        logger.info(f"Using delta range: call={self.delta_range['call']}, put={self.delta_range['put']}")
    
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate Greek Sentiment Analysis features.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - price_column (str): Column name for price
                - call_vega_column (str): Column name for call vega
                - put_vega_column (str): Column name for put vega
                - call_delta_column (str): Column name for call delta
                - put_delta_column (str): Column name for put delta
                - call_theta_column (str): Column name for call theta
                - put_theta_column (str): Column name for put theta
                - strike_column (str): Column name for strike price
                - date_column (str): Column name for date
                - time_column (str): Column name for time
                - expiry_column (str): Column name for expiry date
                - dte_column (str): Column name for DTE
                - specific_dte (int): Specific DTE to use for calculations
            
        Returns:
            pd.DataFrame: Data with calculated Greek Sentiment Analysis features
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Get column names from kwargs or use defaults
        price_column = kwargs.get('price_column', 'Close')
        call_vega_column = kwargs.get('call_vega_column', 'Call_Vega')
        put_vega_column = kwargs.get('put_vega_column', 'Put_Vega')
        call_delta_column = kwargs.get('call_delta_column', 'Call_Delta')
        put_delta_column = kwargs.get('put_delta_column', 'Put_Delta')
        call_theta_column = kwargs.get('call_theta_column', 'Call_Theta')
        put_theta_column = kwargs.get('put_theta_column', 'Put_Theta')
        strike_column = kwargs.get('strike_column', 'Strike')
        date_column = kwargs.get('date_column', 'Date')
        time_column = kwargs.get('time_column', 'Time')
        expiry_column = kwargs.get('expiry_column', 'Expiry')
        dte_column = kwargs.get('dte_column', 'DTE')
        
        # Check if required columns exist
        required_columns = [
            price_column, call_vega_column, put_vega_column,
            call_delta_column, put_delta_column, call_theta_column,
            put_theta_column, strike_column
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return df
        
        # Filter by delta range
        df_calls = df[(df[call_delta_column] >= self.delta_range['call']['min']) & 
                      (df[call_delta_column] <= self.delta_range['call']['max'])]
        
        df_puts = df[(df[put_delta_column] >= self.delta_range['put']['min']) & 
                     (df[put_delta_column] <= self.delta_range['put']['max'])]
        
        logger.info(f"Filtered {len(df_calls)} call options and {len(df_puts)} put options by delta range")
        
        # Process by expiry if expiry column exists
        if expiry_column in df.columns:
            # Get unique expiry dates
            expiry_dates = sorted(df[expiry_column].unique())
            
            if len(expiry_dates) >= 2:
                # Get near and next expiry
                near_expiry = expiry_dates[0]
                next_expiry = expiry_dates[1]
                
                logger.info(f"Processing near expiry: {near_expiry} and next expiry: {next_expiry}")
                
                # Calculate features for near expiry
                near_expiry_data = df[df[expiry_column] == near_expiry]
                near_features = self._calculate_expiry_features(
                    near_expiry_data,
                    price_column,
                    call_vega_column,
                    put_vega_column,
                    call_delta_column,
                    put_delta_column,
                    call_theta_column,
                    put_theta_column,
                    strike_column,
                    date_column,
                    time_column
                )
                
                # Calculate features for next expiry
                next_expiry_data = df[df[expiry_column] == next_expiry]
                next_features = self._calculate_expiry_features(
                    next_expiry_data,
                    price_column,
                    call_vega_column,
                    put_vega_column,
                    call_delta_column,
                    put_delta_column,
                    call_theta_column,
                    put_theta_column,
                    strike_column,
                    date_column,
                    time_column
                )
                
                # Combine features with expiry weights
                combined_features = {}
                
                for key in near_features:
                    if key in next_features:
                        combined_features[key] = (
                            near_features[key] * self.expiry_weights['near'] +
                            next_features[key] * self.expiry_weights['next']
                        )
                    else:
                        combined_features[key] = near_features[key]
                
                # Add combined features to dataframe
                for col, values in combined_features.items():
                    df[col] = values
                
                logger.info(f"Combined features from near and next expiry with weights {self.expiry_weights}")
            else:
                # Process single expiry
                logger.info(f"Processing single expiry: {expiry_dates[0] if expiry_dates else 'unknown'}")
                
                expiry_features = self._calculate_expiry_features(
                    df,
                    price_column,
                    call_vega_column,
                    put_vega_column,
                    call_delta_column,
                    put_delta_column,
                    call_theta_column,
                    put_theta_column,
                    strike_column,
                    date_column,
                    time_column
                )
                
                # Add features to dataframe
                for col, values in expiry_features.items():
                    df[col] = values
        else:
            # Process without expiry information
            logger.info("Processing without expiry information")
            
            expiry_features = self._calculate_expiry_features(
                df,
                price_column,
                call_vega_column,
                put_vega_column,
                call_delta_column,
                put_delta_column,
                call_theta_column,
                put_theta_column,
                strike_column,
                date_column,
                time_column
            )
            
            # Add features to dataframe
            for col, values in expiry_features.items():
                df[col] = values
        
        logger.info(f"Calculated Greek Sentiment Analysis features")
        
        return df
    
    def _calculate_expiry_features(self, expiry_data, price_column, 
                                  call_vega_column, put_vega_column,
                                  call_delta_column, put_delta_column,
                                  call_theta_column, put_theta_column,
                                  strike_column, date_column, time_column):
        """
        Calculate Greek sentiment features for a specific expiry.
        
        Args:
            expiry_data (pd.DataFrame): Data for this expiry
            price_column (str): Column name for price
            call_vega_column (str): Column name for call vega
            put_vega_column (str): Column name for put vega
            call_delta_column (str): Column name for call delta
            put_delta_column (str): Column name for put delta
            call_theta_column (str): Column name for call theta
            put_theta_column (str): Column name for put theta
            strike_column (str): Column name for strike price
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary of feature columns and values
        """
        features = {}
        
        # Step 1: Identify ATM strike
        atm_strikes = self._identify_atm_strikes(expiry_data, price_column, strike_column)
        
        # Step 2: Select strikes to analyze
        selected_strikes = self._select_strikes(expiry_data, atm_strikes, strike_column)
        
        # Step 3: Calculate Vega changes
        vega_changes = self._calculate_vega_changes(expiry_data, selected_strikes, call_vega_column, put_vega_column, strike_column, date_column, time_column)
        features['Vega_Change_Call'] = vega_changes['call']
        features['Vega_Change_Put'] = vega_changes['put']
        features['Vega_Sentiment'] = vega_changes['sentiment']
        
        # Step 4: Calculate Delta changes
        delta_changes = self._calculate_delta_changes(expiry_data, selected_strikes, call_delta_column, put_delta_column, strike_column, date_column, time_column)
        features['Delta_Change_Call'] = delta_changes['call']
        features['Delta_Change_Put'] = delta_changes['put']
        features['Delta_Sentiment'] = delta_changes['sentiment']
        
        # Step 5: Calculate Theta changes
        theta_changes = self._calculate_theta_changes(expiry_data, selected_strikes, call_theta_column, put_theta_column, strike_column, date_column, time_column)
        features['Theta_Change_Call'] = theta_changes['call']
        features['Theta_Change_Put'] = theta_changes['put']
        features['Theta_Sentiment'] = theta_changes['sentiment']
        
        # Step 6: Calculate combined Greek sentiment
        greek_sentiment = self._calculate_greek_sentiment(vega_changes['sentiment'], delta_changes['sentiment'], theta_changes['sentiment'])
        features['Greek_Sentiment_Signal'] = greek_sentiment
        
        # Step 7: Classify sentiment
        sentiment = self._classify_sentiment(greek_sentiment)
        features['Greek_Sentiment'] = sentiment
        
        # Step 8: Calculate Greek sentiment regime
        regime = self._calculate_greek_sentiment_regime(sentiment, vega_changes['sentiment'], delta_changes['sentiment'])
        features['Greek_Sentiment_Regime'] = regime
        
        return features
    
    def _identify_atm_strikes(self, data, price_column, strike_column):
        """
        Identify ATM strikes.
        
        Args:
            data (pd.DataFrame): Input data
            price_column (str): Column name for price
            strike_column (str): Column name for strike price
            
        Returns:
            list: List of ATM strikes
        """
        # Get current price
        current_price = data[price_column].iloc[-1]
        
        # Get unique strikes
        unique_strikes = data[strike_column].unique()
        
        # Sort strikes
        unique_strikes = np.sort(unique_strikes)
        
        # Find closest strike
        closest_strike = unique_strikes[np.abs(unique_strikes - current_price).argmin()]
        
        # Find strikes above and below
        strike_above = unique_strikes[unique_strikes > closest_strike].min() if any(unique_strikes > closest_strike) else closest_strike
        strike_below = unique_strikes[unique_strikes < closest_strike].max() if any(unique_strikes < closest_strike) else closest_strike
        
        # Return ATM strikes
        return [strike_below, closest_strike, strike_above]
    
    def _select_strikes(self, data, atm_strikes, strike_column):
        """
        Select strikes to analyze.
        
        Args:
            data (pd.DataFrame): Input data
            atm_strikes (list): List of ATM strikes
            strike_column (str): Column name for strike price
            
        Returns:
            dict: Dictionary of selected strikes by category
        """
        # Get unique strikes
        unique_strikes = np.sort(data[strike_column].unique())
        
        # Get ATM strike
        atm_strike = atm_strikes[1]
        
        # Find index of ATM strike
        atm_index = np.where(unique_strikes == atm_strike)[0][0]
        
        # Determine number of strikes to select (7 above and 7 below ATM, plus ATM = 15)
        num_above = 7
        num_below = 7
        
        # Select strikes above ATM
        strikes_above = unique_strikes[atm_index+1:atm_index+1+num_above] if atm_index+1+num_above <= len(unique_strikes) else unique_strikes[atm_index+1:]
        
        # Select strikes below ATM
        strikes_below = unique_strikes[max(0, atm_index-num_below):atm_index] if atm_index-num_below >= 0 else unique_strikes[:atm_index]
        
        # Categorize strikes
        selected_strikes = {
            'atm': [atm_strike],
            'near_otm': [],
            'mid_otm': [],
            'far_otm': []
        }
        
        # Add strikes above ATM
        for i, strike in enumerate(strikes_above):
            if i < len(strikes_above) // 3:
                selected_strikes['near_otm'].append(strike)
            elif i < 2 * len(strikes_above) // 3:
                selected_strikes['mid_otm'].append(strike)
            else:
                selected_strikes['far_otm'].append(strike)
        
        # Add strikes below ATM
        for i, strike in enumerate(strikes_below):
            if i >= 2 * len(strikes_below) // 3:
                selected_strikes['near_otm'].append(strike)
            elif i >= len(strikes_below) // 3:
                selected_strikes['mid_otm'].append(strike)
            else:
                selected_strikes['far_otm'].append(strike)
        
        return selected_strikes
    
    def _calculate_vega_changes(self, data, selected_strikes, call_vega_column, put_vega_column, strike_column, date_column, time_column):
        """
        Calculate Vega changes.
        
        Args:
            data (pd.DataFrame): Input data
            selected_strikes (dict): Dictionary of selected strikes by category
            call_vega_column (str): Column name for call vega
            put_vega_column (str): Column name for put vega
            strike_column (str): Column name for strike price
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with call, put, and sentiment values
        """
        # Initialize weighted Vega changes
        weighted_call_vega_change = 0
        weighted_put_vega_change = 0
        
        # Check if date and time columns exist
        has_datetime = date_column in data.columns and time_column in data.columns
        
        # Process each strike category
        for category, strikes in selected_strikes.items():
            # Skip if no strikes in this category
            if not strikes:
                continue
            
            # Get weight for this category
            category_weight = self.strike_weights.get(category, 0.1)
            
            # Filter data for these strikes
            category_data = data[data[strike_column].isin(strikes)]
            
            # Skip if no data for these strikes
            if category_data.empty:
                continue
            
            # Calculate Vega changes
            if has_datetime:
                # Group by date and time
                grouped = category_data.groupby([date_column, time_column])
                
                # Get first and last groups
                if len(grouped) >= 2:
                    first_group = grouped.first()
                    last_group = grouped.last()
                    
                    # Calculate changes
                    call_vega_change = (last_group[call_vega_column].mean() - first_group[call_vega_column].mean()) / first_group[call_vega_column].mean() if first_group[call_vega_column].mean() != 0 else 0
                    put_vega_change = (last_group[put_vega_column].mean() - first_group[put_vega_column].mean()) / first_group[put_vega_column].mean() if first_group[put_vega_column].mean() != 0 else 0
                else:
                    call_vega_change = 0
                    put_vega_change = 0
            else:
                # Calculate changes from first to last row
                if len(category_data) >= 2:
                    call_vega_first = category_data[call_vega_column].iloc[0]
                    call_vega_last = category_data[call_vega_column].iloc[-1]
                    put_vega_first = category_data[put_vega_column].iloc[0]
                    put_vega_last = category_data[put_vega_column].iloc[-1]
                    
                    call_vega_change = (call_vega_last - call_vega_first) / call_vega_first if call_vega_first != 0 else 0
                    put_vega_change = (put_vega_last - put_vega_first) / put_vega_first if put_vega_first != 0 else 0
                else:
                    call_vega_change = 0
                    put_vega_change = 0
            
            # Add weighted changes
            weighted_call_vega_change += call_vega_change * category_weight
            weighted_put_vega_change += put_vega_change * category_weight
        
        # Calculate Vega sentiment
        # Positive call vega change is bullish, positive put vega change is bearish
        vega_sentiment = weighted_call_vega_change - weighted_put_vega_change
        
        return {
            'call': weighted_call_vega_change,
            'put': weighted_put_vega_change,
            'sentiment': vega_sentiment
        }
    
    def _calculate_delta_changes(self, data, selected_strikes, call_delta_column, put_delta_column, strike_column, date_column, time_column):
        """
        Calculate Delta changes.
        
        Args:
            data (pd.DataFrame): Input data
            selected_strikes (dict): Dictionary of selected strikes by category
            call_delta_column (str): Column name for call delta
            put_delta_column (str): Column name for put delta
            strike_column (str): Column name for strike price
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with call, put, and sentiment values
        """
        # Initialize weighted Delta changes
        weighted_call_delta_change = 0
        weighted_put_delta_change = 0
        
        # Check if date and time columns exist
        has_datetime = date_column in data.columns and time_column in data.columns
        
        # Process each strike category
        for category, strikes in selected_strikes.items():
            # Skip if no strikes in this category
            if not strikes:
                continue
            
            # Get weight for this category
            category_weight = self.strike_weights.get(category, 0.1)
            
            # Filter data for these strikes
            category_data = data[data[strike_column].isin(strikes)]
            
            # Skip if no data for these strikes
            if category_data.empty:
                continue
            
            # Calculate Delta changes
            if has_datetime:
                # Group by date and time
                grouped = category_data.groupby([date_column, time_column])
                
                # Get first and last groups
                if len(grouped) >= 2:
                    first_group = grouped.first()
                    last_group = grouped.last()
                    
                    # Calculate changes
                    call_delta_change = (last_group[call_delta_column].mean() - first_group[call_delta_column].mean()) / first_group[call_delta_column].mean() if first_group[call_delta_column].mean() != 0 else 0
                    put_delta_change = (last_group[put_delta_column].mean() - first_group[put_delta_column].mean()) / first_group[put_delta_column].mean() if first_group[put_delta_column].mean() != 0 else 0
                else:
                    call_delta_change = 0
                    put_delta_change = 0
            else:
                # Calculate changes from first to last row
                if len(category_data) >= 2:
                    call_delta_first = category_data[call_delta_column].iloc[0]
                    call_delta_last = category_data[call_delta_column].iloc[-1]
                    put_delta_first = category_data[put_delta_column].iloc[0]
                    put_delta_last = category_data[put_delta_column].iloc[-1]
                    
                    call_delta_change = (call_delta_last - call_delta_first) / call_delta_first if call_delta_first != 0 else 0
                    put_delta_change = (put_delta_last - put_delta_first) / put_delta_first if put_delta_first != 0 else 0
                else:
                    call_delta_change = 0
                    put_delta_change = 0
            
            # Add weighted changes
            weighted_call_delta_change += call_delta_change * category_weight
            weighted_put_delta_change += put_delta_change * category_weight
        
        # Calculate Delta sentiment
        # Positive call delta change is bullish, positive put delta change is bearish
        delta_sentiment = weighted_call_delta_change - weighted_put_delta_change
        
        return {
            'call': weighted_call_delta_change,
            'put': weighted_put_delta_change,
            'sentiment': delta_sentiment
        }
    
    def _calculate_theta_changes(self, data, selected_strikes, call_theta_column, put_theta_column, strike_column, date_column, time_column):
        """
        Calculate Theta changes.
        
        Args:
            data (pd.DataFrame): Input data
            selected_strikes (dict): Dictionary of selected strikes by category
            call_theta_column (str): Column name for call theta
            put_theta_column (str): Column name for put theta
            strike_column (str): Column name for strike price
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with call, put, and sentiment values
        """
        # Initialize weighted Theta changes
        weighted_call_theta_change = 0
        weighted_put_theta_change = 0
        
        # Check if date and time columns exist
        has_datetime = date_column in data.columns and time_column in data.columns
        
        # Process each strike category
        for category, strikes in selected_strikes.items():
            # Skip if no strikes in this category
            if not strikes:
                continue
            
            # Get weight for this category
            category_weight = self.strike_weights.get(category, 0.1)
            
            # Filter data for these strikes
            category_data = data[data[strike_column].isin(strikes)]
            
            # Skip if no data for these strikes
            if category_data.empty:
                continue
            
            # Calculate Theta changes
            if has_datetime:
                # Group by date and time
                grouped = category_data.groupby([date_column, time_column])
                
                # Get first and last groups
                if len(grouped) >= 2:
                    first_group = grouped.first()
                    last_group = grouped.last()
                    
                    # Calculate changes
                    call_theta_change = (last_group[call_theta_column].mean() - first_group[call_theta_column].mean()) / first_group[call_theta_column].mean() if first_group[call_theta_column].mean() != 0 else 0
                    put_theta_change = (last_group[put_theta_column].mean() - first_group[put_theta_column].mean()) / first_group[put_theta_column].mean() if first_group[put_theta_column].mean() != 0 else 0
                else:
                    call_theta_change = 0
                    put_theta_change = 0
            else:
                # Calculate changes from first to last row
                if len(category_data) >= 2:
                    call_theta_first = category_data[call_theta_column].iloc[0]
                    call_theta_last = category_data[call_theta_column].iloc[-1]
                    put_theta_first = category_data[put_theta_column].iloc[0]
                    put_theta_last = category_data[put_theta_column].iloc[-1]
                    
                    call_theta_change = (call_theta_last - call_theta_first) / call_theta_first if call_theta_first != 0 else 0
                    put_theta_change = (put_theta_last - put_theta_first) / put_theta_first if put_theta_first != 0 else 0
                else:
                    call_theta_change = 0
                    put_theta_change = 0
            
            # Add weighted changes
            weighted_call_theta_change += call_theta_change * category_weight
            weighted_put_theta_change += put_theta_change * category_weight
        
        # Calculate Theta sentiment
        # Negative call theta change is bullish, negative put theta change is bearish
        theta_sentiment = -weighted_call_theta_change + weighted_put_theta_change
        
        return {
            'call': weighted_call_theta_change,
            'put': weighted_put_theta_change,
            'sentiment': theta_sentiment
        }
    
    def _calculate_greek_sentiment(self, vega_sentiment, delta_sentiment, theta_sentiment):
        """
        Calculate combined Greek sentiment.
        
        Args:
            vega_sentiment (float): Vega sentiment
            delta_sentiment (float): Delta sentiment
            theta_sentiment (float): Theta sentiment
            
        Returns:
            float: Combined Greek sentiment
        """
        # Calculate weighted sentiment
        greek_sentiment = (
            vega_sentiment * self.greek_weights['vega'] +
            delta_sentiment * self.greek_weights['delta'] +
            theta_sentiment * self.greek_weights['theta']
        )
        
        return greek_sentiment
    
    def _classify_sentiment(self, greek_sentiment):
        """
        Classify Greek sentiment.
        
        Args:
            greek_sentiment (float): Greek sentiment value
            
        Returns:
            str: Sentiment classification
        """
        if greek_sentiment >= self.sentiment_thresholds['strong_bullish']:
            return 'Strong_Bullish'
        elif greek_sentiment >= self.sentiment_thresholds['mild_bullish']:
            return 'Mild_Bullish'
        elif greek_sentiment <= self.sentiment_thresholds['strong_bearish']:
            return 'Strong_Bearish'
        elif greek_sentiment <= self.sentiment_thresholds['mild_bearish']:
            return 'Mild_Bearish'
        else:
            return 'Neutral'
    
    def _calculate_greek_sentiment_regime(self, sentiment, vega_sentiment, delta_sentiment):
        """
        Calculate Greek sentiment regime.
        
        Args:
            sentiment (str): Sentiment classification
            vega_sentiment (float): Vega sentiment
            delta_sentiment (float): Delta sentiment
            
        Returns:
            str: Greek sentiment regime
        """
        # Determine regime based on sentiment and confirmation
        if sentiment == 'Strong_Bullish' and vega_sentiment > 0 and delta_sentiment > 0:
            return 'Strong_Bullish_Confirmed'
        elif sentiment == 'Strong_Bullish':
            return 'Strong_Bullish_Unconfirmed'
        elif sentiment == 'Mild_Bullish' and vega_sentiment > 0:
            return 'Mild_Bullish_Confirmed'
        elif sentiment == 'Mild_Bullish':
            return 'Mild_Bullish_Unconfirmed'
        elif sentiment == 'Strong_Bearish' and vega_sentiment < 0 and delta_sentiment < 0:
            return 'Strong_Bearish_Confirmed'
        elif sentiment == 'Strong_Bearish':
            return 'Strong_Bearish_Unconfirmed'
        elif sentiment == 'Mild_Bearish' and vega_sentiment < 0:
            return 'Mild_Bearish_Confirmed'
        elif sentiment == 'Mild_Bearish':
            return 'Mild_Bearish_Unconfirmed'
        else:
            return 'Neutral'

# Function to calculate Greek sentiment (for backward compatibility)
def calculate_greek_sentiment(market_data, config=None):
    """
    Calculate Greek sentiment based on market data.
    
    Args:
        market_data (DataFrame): Market data
        config (dict): Configuration settings
        
    Returns:
        Series: Greek sentiment values
    """
    logger.info("Calculating Greek sentiment")
    
    try:
        # Create Greek sentiment calculator
        calculator = GreekSentimentAnalysis(config)
        
        # Calculate Greek sentiment features
        result_df = calculator.calculate_features(market_data)
        
        # Return Greek sentiment series
        if 'Greek_Sentiment' in result_df.columns:
            return result_df['Greek_Sentiment']
        else:
            logger.warning("Greek_Sentiment column not found in result")
            return None
    
    except Exception as e:
        logger.error(f"Error calculating Greek sentiment: {str(e)}")
        return None
