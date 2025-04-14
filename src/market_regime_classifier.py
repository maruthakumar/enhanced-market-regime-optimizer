    def _detect_transitions(self, data, directional_column, volatility_column, date_column, time_column):
        """
        Detect regime transitions.
        
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
        
        # Check if we have enough data for transition detection
        if len(data) <= self.transition_lookback:
            return {
                'type': transition_type,
                'probability': transition_probability
            }
        
        # Check if we have datetime columns
        has_datetime = date_column in data.columns and time_column in data.columns
        
        if has_datetime:
            # Sort by datetime
            data = data.sort_values([date_column, time_column])
        
        # Calculate directional change
        directional_change = data[directional_column].diff(self.transition_lookback)
        
        # Calculate volatility change
        volatility_change = data[volatility_column].diff(self.transition_lookback)
        
        # Detect bullish to bearish transition
        bullish_to_bearish = (directional_change < -self.transition_threshold)
        
        # Detect bearish to bullish transition
        bearish_to_bullish = (directional_change > self.transition_threshold)
        
        # Detect volatility expansion
        volatility_expansion = (volatility_change > self.transition_threshold)
        
        # Set transition type and probability
        transition_type[bullish_to_bearish] = 'Bullish_To_Bearish'
        transition_type[bearish_to_bullish] = 'Bearish_To_Bullish'
        transition_type[volatility_expansion] = 'Volatility_Expansion'
        
        # Calculate transition probability
        transition_probability[bullish_to_bearish] = np.abs(directional_change[bullish_to_bearish])
        transition_probability[bearish_to_bullish] = np.abs(directional_change[bearish_to_bullish])
        transition_probability[volatility_expansion] = np.abs(volatility_change[volatility_expansion])
        
        # Normalize probability to [0, 1]
        transition_probability = transition_probability.clip(0, 1)
        
        return {
            'type': transition_type,
            'probability': transition_probability
        }
