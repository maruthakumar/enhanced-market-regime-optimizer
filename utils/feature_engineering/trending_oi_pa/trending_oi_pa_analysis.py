"""
Consolidated Trending OI with PA Analysis Module

This module implements Trending Open Interest with Price Action analysis,
analyzing ATM plus 7 strikes above and 7 strikes below (total of 15 strikes)
and implementing rolling calculation for trending OI of calls and puts.

Features:
- OI trends analysis
- OI accumulation detection
- Price momentum relative to OI changes
- Breakout/breakdown confirmation with OI
- Support/resistance tests with OI confirmation
- Divergence/convergence between OI and price
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger(__name__)

class TrendingOIWithPAAnalysis:
    """
    Consolidated Trending OI with PA Analysis.
    
    This class implements analysis of Open Interest trends and their relationship
    with Price Action, focusing on ATM plus 7 strikes above and 7 strikes below
    (total of 15 strikes) and implementing rolling calculation for trending OI
    of calls and puts.
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
        
        logger.info(f"Initialized Trending OI with PA Analysis with default weight {self.default_weight}")
        logger.info(f"Using {self.strikes_above_atm} strikes above ATM and {self.strikes_below_atm} strikes below ATM")
    
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate Trending OI with PA Analysis features.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
                - price_column (str): Column name for price
                - call_oi_column (str): Column name for call open interest
                - put_oi_column (str): Column name for put open interest
                - volume_column (str): Column name for volume
                - strike_column (str): Column name for strike price
                - date_column (str): Column name for date
                - time_column (str): Column name for time
                - expiry_column (str): Column name for expiry date
                - dte_column (str): Column name for DTE
                - specific_dte (int): Specific DTE to use for calculations
            
        Returns:
            pd.DataFrame: Data with calculated Trending OI with PA Analysis features
        """
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Get column names from kwargs or use defaults
        price_column = kwargs.get('price_column', 'Close')
        call_oi_column = kwargs.get('call_oi_column', 'Call_OI')
        put_oi_column = kwargs.get('put_oi_column', 'Put_OI')
        volume_column = kwargs.get('volume_column', 'Volume')
        strike_column = kwargs.get('strike_column', 'Strike')
        date_column = kwargs.get('date_column', 'Date')
        time_column = kwargs.get('time_column', 'Time')
        expiry_column = kwargs.get('expiry_column', 'Expiry')
        dte_column = kwargs.get('dte_column', 'DTE')
        
        # Check if required columns exist
        required_columns = [
            price_column, call_oi_column, put_oi_column,
            volume_column, strike_column
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return df
        
        # Step 1: Identify ATM strike
        atm_strike = self._identify_atm_strike(df, price_column, strike_column)
        
        # Step 2: Select strikes to analyze
        selected_strikes = self._select_strikes(df, atm_strike, strike_column)
        
        # Step 3: Calculate OI trends
        oi_trends = self._calculate_oi_trends(df, selected_strikes, call_oi_column, put_oi_column, strike_column, date_column, time_column)
        df['Call_OI_Trend'] = oi_trends['call_trend']
        df['Put_OI_Trend'] = oi_trends['put_trend']
        df['Net_OI_Trend'] = oi_trends['net_trend']
        
        # Step 4: Calculate rolling OI trends
        rolling_oi_trends = self._calculate_rolling_oi_trends(df, selected_strikes, call_oi_column, put_oi_column, strike_column, date_column, time_column)
        df['Rolling_Call_OI_Trend'] = rolling_oi_trends['rolling_call_trend']
        df['Rolling_Put_OI_Trend'] = rolling_oi_trends['rolling_put_trend']
        df['Rolling_Net_OI_Trend'] = rolling_oi_trends['rolling_net_trend']
        
        # Step 5: Calculate OI-Price relationship
        oi_price_relationship = self._calculate_oi_price_relationship(df, price_column, 'Call_OI_Trend', 'Put_OI_Trend', 'Net_OI_Trend')
        df['OI_Price_Relationship'] = oi_price_relationship['relationship']
        df['OI_Price_Signal'] = oi_price_relationship['signal']
        
        # Step 6: Calculate OI-Price divergence/convergence
        oi_price_divergence = self._calculate_oi_price_divergence(df, price_column, 'Rolling_Call_OI_Trend', 'Rolling_Put_OI_Trend', 'Rolling_Net_OI_Trend')
        df['OI_Price_Divergence'] = oi_price_divergence['divergence']
        df['OI_Price_Divergence_Signal'] = oi_price_divergence['signal']
        
        # Step 7: Calculate OI trend strength
        oi_trend_strength = self._calculate_oi_trend_strength(df, 'Rolling_Call_OI_Trend', 'Rolling_Put_OI_Trend', 'Rolling_Net_OI_Trend')
        df['OI_Trend_Strength'] = oi_trend_strength['strength']
        df['OI_Trend_Strength_Signal'] = oi_trend_strength['signal']
        
        # Step 8: Calculate combined OI-PA signal
        combined_signal = self._calculate_combined_signal(df, 'OI_Price_Signal', 'OI_Price_Divergence_Signal', 'OI_Trend_Strength_Signal')
        df['OI_PA_Signal'] = combined_signal
        
        # Step 9: Calculate OI-PA regime
        oi_pa_regime = self._calculate_oi_pa_regime(df, 'OI_PA_Signal', 'OI_Trend_Strength', 'OI_Price_Relationship')
        df['OI_PA_Regime'] = oi_pa_regime
        
        logger.info(f"Calculated Trending OI with PA Analysis features")
        
        return df
    
    def _identify_atm_strike(self, data, price_column, strike_column):
        """
        Identify ATM strike.
        
        Args:
            data (pd.DataFrame): Input data
            price_column (str): Column name for price
            strike_column (str): Column name for strike price
            
        Returns:
            float: ATM strike price
        """
        # Get current price
        current_price = data[price_column].iloc[-1]
        
        # Get unique strikes
        unique_strikes = data[strike_column].unique()
        
        # Sort strikes
        unique_strikes = np.sort(unique_strikes)
        
        # Find closest strike
        closest_strike = unique_strikes[np.abs(unique_strikes - current_price).argmin()]
        
        return closest_strike
    
    def _select_strikes(self, data, atm_strike, strike_column):
        """
        Select strikes to analyze.
        
        Args:
            data (pd.DataFrame): Input data
            atm_strike (float): ATM strike price
            strike_column (str): Column name for strike price
            
        Returns:
            list: List of selected strikes
        """
        # Get unique strikes
        unique_strikes = np.sort(data[strike_column].unique())
        
        # Find index of ATM strike
        atm_index = np.where(unique_strikes == atm_strike)[0][0]
        
        # Select strikes above ATM
        strikes_above = unique_strikes[atm_index+1:atm_index+1+self.strikes_above_atm] if atm_index+1+self.strikes_above_atm <= len(unique_strikes) else unique_strikes[atm_index+1:]
        
        # Select strikes below ATM
        strikes_below = unique_strikes[max(0, atm_index-self.strikes_below_atm):atm_index] if atm_index-self.strikes_below_atm >= 0 else unique_strikes[:atm_index]
        
        # Combine strikes
        selected_strikes = list(strikes_below) + [atm_strike] + list(strikes_above)
        
        return selected_strikes
    
    def _calculate_oi_trends(self, data, selected_strikes, call_oi_column, put_oi_column, strike_column, date_column, time_column):
        """
        Calculate OI trends.
        
        Args:
            data (pd.DataFrame): Input data
            selected_strikes (list): List of selected strikes
            call_oi_column (str): Column name for call open interest
            put_oi_column (str): Column name for put open interest
            strike_column (str): Column name for strike price
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with call, put, and net trend values
        """
        # Filter data for selected strikes
        filtered_data = data[data[strike_column].isin(selected_strikes)]
        
        # Check if date and time columns exist
        has_datetime = date_column in data.columns and time_column in data.columns
        
        # Calculate OI trends
        if has_datetime:
            # Group by date and time
            grouped = filtered_data.groupby([date_column, time_column])
            
            # Calculate call OI trend
            call_oi_sum = grouped[call_oi_column].sum()
            call_oi_pct_change = call_oi_sum.pct_change()
            
            # Calculate put OI trend
            put_oi_sum = grouped[put_oi_column].sum()
            put_oi_pct_change = put_oi_sum.pct_change()
            
            # Calculate net OI trend
            net_oi_sum = call_oi_sum - put_oi_sum
            net_oi_pct_change = net_oi_sum.pct_change()
            
            # Reindex to match original data
            call_trend = pd.Series(index=data.index)
            put_trend = pd.Series(index=data.index)
            net_trend = pd.Series(index=data.index)
            
            for (date, time), group in data.groupby([date_column, time_column]):
                if (date, time) in call_oi_pct_change.index:
                    call_trend.loc[group.index] = call_oi_pct_change.loc[(date, time)]
                    put_trend.loc[group.index] = put_oi_pct_change.loc[(date, time)]
                    net_trend.loc[group.index] = net_oi_pct_change.loc[(date, time)]
        else:
            # Calculate trends without datetime grouping
            call_oi_sum = filtered_data.groupby(strike_column)[call_oi_column].sum()
            put_oi_sum = filtered_data.groupby(strike_column)[put_oi_column].sum()
            
            # Calculate pct change
            call_trend = call_oi_sum.pct_change().mean()
            put_trend = put_oi_sum.pct_change().mean()
            net_trend = (call_oi_sum - put_oi_sum).pct_change().mean()
            
            # Create series with same value for all rows
            call_trend = pd.Series(call_trend, index=data.index)
            put_trend = pd.Series(put_trend, index=data.index)
            net_trend = pd.Series(net_trend, index=data.index)
        
        return {
            'call_trend': call_trend,
            'put_trend': put_trend,
            'net_trend': net_trend
        }
    
    def _calculate_rolling_oi_trends(self, data, selected_strikes, call_oi_column, put_oi_column, strike_column, date_column, time_column):
        """
        Calculate rolling OI trends.
        
        Args:
            data (pd.DataFrame): Input data
            selected_strikes (list): List of selected strikes
            call_oi_column (str): Column name for call open interest
            put_oi_column (str): Column name for put open interest
            strike_column (str): Column name for strike price
            date_column (str): Column name for date
            time_column (str): Column name for time
            
        Returns:
            dict: Dictionary with rolling call, put, and net trend values
        """
        # Filter data for selected strikes
        filtered_data = data[data[strike_column].isin(selected_strikes)]
        
        # Check if date and time columns exist
        has_datetime = date_column in data.columns and time_column in data.columns
        
        # Calculate rolling OI trends
        if has_datetime:
            # Group by date and time
            grouped = filtered_data.groupby([date_column, time_column])
            
            # Calculate call OI trend
            call_oi_sum = grouped[call_oi_column].sum()
            rolling_call_oi = call_oi_sum.rolling(window=self.short_window, min_periods=1)
            rolling_call_oi_pct_change = (call_oi_sum - rolling_call_oi.mean()) / rolling_call_oi.mean()
            
            # Calculate put OI trend
            put_oi_sum = grouped[put_oi_column].sum()
            rolling_put_oi = put_oi_sum.rolling(window=self.short_window, min_periods=1)
            rolling_put_oi_pct_change = (put_oi_sum - rolling_put_oi.mean()) / rolling_put_oi.mean()
            
            # Calculate net OI trend
            net_oi_sum = call_oi_sum - put_oi_sum
            rolling_net_oi = net_oi_sum.rolling(window=self.short_window, min_periods=1)
            rolling_net_oi_pct_change = (net_oi_sum - rolling_net_oi.mean()) / rolling_net_oi.mean()
            
            # Reindex to match original data
            rolling_call_trend = pd.Series(index=data.index)
            rolling_put_trend = pd.Series(index=data.index)
            rolling_net_trend = pd.Series(index=data.index)
            
            for (date, time), group in data.groupby([date_column, time_column]):
                if (date, time) in rolling_call_oi_pct_change.index:
                    rolling_call_trend.loc[group.index] = rolling_call_oi_pct_change.loc[(date, time)]
                    rolling_put_trend.loc[group.index] = rolling_put_oi_pct_change.loc[(date, time)]
                    rolling_net_trend.loc[group.index] = rolling_net_oi_pct_change.loc[(date, time)]
        else:
            # Calculate trends without datetime grouping
            call_oi_sum = filtered_data.groupby(strike_column)[call_oi_column].sum()
            put_oi_sum = filtered_data.groupby(strike_column)[put_oi_column].sum()
            
            # Calculate rolling means
            rolling_call_oi = call_oi_sum.rolling(window=self.short_window, min_periods=1)
            rolling_put_oi = put_oi_sum.rolling(window=self.short_window, min_periods=1)
            rolling_net_oi = (call_oi_sum - put_oi_sum).rolling(window=self.short_window, min_periods=1)
            
            # Calculate pct change
            rolling_call_trend = ((call_oi_sum - rolling_call_oi.mean()) / rolling_call_oi.mean()).mean()
            rolling_put_trend = ((put_oi_sum - rolling_put_oi.mean()) / rolling_put_oi.mean()).mean()
            rolling_net_trend = ((call_oi_sum - put_oi_sum - rolling_net_oi.mean()) / rolling_net_oi.mean()).mean()
            
            # Create series with same value for all rows
            rolling_call_trend = pd.Series(rolling_call_trend, index=data.index)
            rolling_put_trend = pd.Series(rolling_put_trend, index=data.index)
            rolling_net_trend = pd.Series(rolling_net_trend, index=data.index)
        
        return {
            'rolling_call_trend': rolling_call_trend,
            'rolling_put_trend': rolling_put_trend,
            'rolling_net_trend': rolling_net_trend
        }
    
    def _calculate_oi_price_relationship(self, data, price_column, call_trend_column, put_trend_column, net_trend_column):
        """
        Calculate OI-Price relationship.
        
        Args:
            data (pd.DataFrame): Input data
            price_column (str): Column name for price
            call_trend_column (str): Column name for call OI trend
            put_trend_column (str): Column name for put OI trend
            net_trend_column (str): Column name for net OI trend
            
        Returns:
            dict: Dictionary with relationship and signal values
        """
        # Calculate price change
        price_change = data[price_column].pct_change()
        
        # Initialize relationship and signal series
        relationship = pd.Series(index=data.index)
        signal = pd.Series(index=data.index)
        
        # Calculate relationship and signal for each row
        for i in range(1, len(data)):
            # Get values
            price_chg = price_change.iloc[i]
            call_trend = data[call_trend_column].iloc[i]
            put_trend = data[put_trend_column].iloc[i]
            net_trend = data[net_trend_column].iloc[i]
            
            # Determine relationship
            if price_chg > self.price_increase_threshold and call_trend > self.oi_increase_threshold:
                relationship.iloc[i] = 'Bullish_Confirmation'
                signal.iloc[i] = 1
            elif price_chg < self.price_decrease_threshold and put_trend > self.oi_increase_threshold:
                relationship.iloc[i] = 'Bearish_Confirmation'
                signal.iloc[i] = -1
            elif price_chg > self.price_increase_threshold and put_trend > self.oi_increase_threshold:
                relationship.iloc[i] = 'Bullish_Divergence'
                signal.iloc[i] = 0.5
            elif price_chg < self.price_decrease_threshold and call_trend > self.oi_increase_threshold:
                relationship.iloc[i] = 'Bearish_Divergence'
                signal.iloc[i] = -0.5
            elif net_trend > self.oi_increase_threshold:
                relationship.iloc[i] = 'Bullish_OI'
                signal.iloc[i] = 0.25
            elif net_trend < self.oi_decrease_threshold:
                relationship.iloc[i] = 'Bearish_OI'
                signal.iloc[i] = -0.25
            else:
                relationship.iloc[i] = 'Neutral'
                signal.iloc[i] = 0
        
        return {
            'relationship': relationship,
            'signal': signal
        }
    
    def _calculate_oi_price_divergence(self, data, price_column, rolling_call_trend_column, rolling_put_trend_column, rolling_net_trend_column):
        """
        Calculate OI-Price divergence/convergence.
        
        Args:
            data (pd.DataFrame): Input data
            price_column (str): Column name for price
            rolling_call_trend_column (str): Column name for rolling call OI trend
            rolling_put_trend_column (str): Column name for rolling put OI trend
            rolling_net_trend_column (str): Column name for rolling net OI trend
            
        Returns:
            dict: Dictionary with divergence and signal values
        """
        # Calculate price change
        price_change = data[price_column].pct_change()
        
        # Calculate rolling price change
        rolling_price_change = price_change.rolling(window=self.short_window, min_periods=1).mean()
        
        # Initialize divergence and signal series
        divergence = pd.Series(index=data.index)
        signal = pd.Series(index=data.index)
        
        # Calculate divergence and signal for each row
        for i in range(self.short_window, len(data)):
            # Get values
            price_chg = rolling_price_change.iloc[i]
            net_trend = data[rolling_net_trend_column].iloc[i]
            
            # Determine divergence
            if price_chg > self.price_increase_threshold and net_trend < 0:
                divergence.iloc[i] = 'Bearish_Divergence'
                signal.iloc[i] = -0.75
            elif price_chg < self.price_decrease_threshold and net_trend > 0:
                divergence.iloc[i] = 'Bullish_Divergence'
                signal.iloc[i] = 0.75
            elif price_chg > self.price_increase_threshold and net_trend > 0:
                divergence.iloc[i] = 'Bullish_Convergence'
                signal.iloc[i] = 1
            elif price_chg < self.price_decrease_threshold and net_trend < 0:
                divergence.iloc[i] = 'Bearish_Convergence'
                signal.iloc[i] = -1
            else:
                divergence.iloc[i] = 'Neutral'
                signal.iloc[i] = 0
        
        return {
            'divergence': divergence,
            'signal': signal
        }
    
    def _calculate_oi_trend_strength(self, data, rolling_call_trend_column, rolling_put_trend_column, rolling_net_trend_column):
        """
        Calculate OI trend strength.
        
        Args:
            data (pd.DataFrame): Input data
            rolling_call_trend_column (str): Column name for rolling call OI trend
            rolling_put_trend_column (str): Column name for rolling put OI trend
            rolling_net_trend_column (str): Column name for rolling net OI trend
            
        Returns:
            dict: Dictionary with strength and signal values
        """
        # Initialize strength and signal series
        strength = pd.Series(index=data.index)
        signal = pd.Series(index=data.index)
        
        # Calculate strength and signal for each row
        for i in range(self.short_window, len(data)):
            # Get values
            call_trend = data[rolling_call_trend_column].iloc[i]
            put_trend = data[rolling_put_trend_column].iloc[i]
            net_trend = data[rolling_net_trend_column].iloc[i]
            
            # Determine strength
            if net_trend > self.strong_trend_threshold:
                strength.iloc[i] = 'Strong_Bullish'
                signal.iloc[i] = 1
            elif net_trend > self.weak_trend_threshold:
                strength.iloc[i] = 'Weak_Bullish'
                signal.iloc[i] = 0.5
            elif net_trend < -self.strong_trend_threshold:
                strength.iloc[i] = 'Strong_Bearish'
                signal.iloc[i] = -1
            elif net_trend < -self.weak_trend_threshold:
                strength.iloc[i] = 'Weak_Bearish'
                signal.iloc[i] = -0.5
            else:
                strength.iloc[i] = 'Neutral'
                signal.iloc[i] = 0
        
        return {
            'strength': strength,
            'signal': signal
        }
    
    def _calculate_combined_signal(self, data, oi_price_signal_column, oi_price_divergence_signal_column, oi_trend_strength_signal_column):
        """
        Calculate combined OI-PA signal.
        
        Args:
            data (pd.DataFrame): Input data
            oi_price_signal_column (str): Column name for OI-Price signal
            oi_price_divergence_signal_column (str): Column name for OI-Price divergence signal
            oi_trend_strength_signal_column (str): Column name for OI trend strength signal
            
        Returns:
            pd.Series: Combined signal
        """
        # Initialize combined signal series
        combined_signal = pd.Series(index=data.index)
        
        # Calculate combined signal for each row
        for i in range(self.short_window, len(data)):
            # Get values
            oi_price_signal = data[oi_price_signal_column].iloc[i]
            oi_price_divergence_signal = data[oi_price_divergence_signal_column].iloc[i]
            oi_trend_strength_signal = data[oi_trend_strength_signal_column].iloc[i]
            
            # Calculate weighted average
            combined_signal.iloc[i] = (
                oi_price_signal * 0.4 +
                oi_price_divergence_signal * 0.3 +
                oi_trend_strength_signal * 0.3
            )
        
        return combined_signal
    
    def _calculate_oi_pa_regime(self, data, combined_signal_column, oi_trend_strength_column, oi_price_relationship_column):
        """
        Calculate OI-PA regime.
        
        Args:
            data (pd.DataFrame): Input data
            combined_signal_column (str): Column name for combined signal
            oi_trend_strength_column (str): Column name for OI trend strength
            oi_price_relationship_column (str): Column name for OI-Price relationship
            
        Returns:
            pd.Series: OI-PA regime
        """
        # Initialize regime series
        regime = pd.Series(index=data.index)
        
        # Calculate regime for each row
        for i in range(self.short_window, len(data)):
            # Get values
            combined_signal = data[combined_signal_column].iloc[i]
            oi_trend_strength = data[oi_trend_strength_column].iloc[i]
            oi_price_relationship = data[oi_price_relationship_column].iloc[i]
            
            # Determine regime
            if combined_signal > 0.75:
                regime.iloc[i] = 'Strong_Bullish'
            elif combined_signal > 0.25:
                regime.iloc[i] = 'Mild_Bullish'
            elif combined_signal < -0.75:
                regime.iloc[i] = 'Strong_Bearish'
            elif combined_signal < -0.25:
                regime.iloc[i] = 'Mild_Bearish'
            else:
                regime.iloc[i] = 'Neutral'
            
            # Add confirmation if available
            if pd.notna(oi_trend_strength) and pd.notna(oi_price_relationship):
                if 'Bullish' in regime.iloc[i] and ('Bullish' in oi_trend_strength or 'Bullish' in oi_price_relationship):
                    regime.iloc[i] += '_Confirmed'
                elif 'Bearish' in regime.iloc[i] and ('Bearish' in oi_trend_strength or 'Bearish' in oi_price_relationship):
                    regime.iloc[i] += '_Confirmed'
        
        return regime

# Function to calculate trending OI with PA (for backward compatibility)
def calculate_trending_oi_pa(market_data, config=None):
    """
    Calculate trending OI with PA based on market data.
    
    Args:
        market_data (DataFrame): Market data
        config (dict): Configuration settings
        
    Returns:
        Series: Trending OI with PA values
    """
    logger.info("Calculating trending OI with PA")
    
    try:
        # Create trending OI with PA calculator
        calculator = TrendingOIWithPAAnalysis(config)
        
        # Calculate trending OI with PA features
        result_df = calculator.calculate_features(market_data)
        
        # Return trending OI with PA series
        if 'OI_PA_Regime' in result_df.columns:
            return result_df['OI_PA_Regime']
        else:
            logger.warning("OI_PA_Regime column not found in result")
            return None
    
    except Exception as e:
        logger.error(f"Error calculating trending OI with PA: {str(e)}")
        return None
