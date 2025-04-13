import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config_manager import ConfigManager

class IndicatorFactory:
    """
    Factory class for creating and configuring technical indicators based on configuration.
    Works with the INI-based configuration system.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the indicator factory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
    def create_indicators(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create all enabled technical indicators based on configuration.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Dictionary of indicator DataFrames
        """
        indicators = {}
        tech_indicators_config = self.config_manager.get_config()['technical_indicators']
        
        # Process each indicator type if enabled
        for indicator_name, indicator_config in tech_indicators_config.items():
            if indicator_config.get('enabled', False):
                self.logger.info(f"Creating indicator: {indicator_name}")
                
                if indicator_name == 'ema_indicators':
                    indicators[indicator_name] = self._create_ema_indicators(data, indicator_config)
                elif indicator_name == 'atr_indicators':
                    indicators[indicator_name] = self._create_atr_indicators(data, indicator_config)
                elif indicator_name == 'iv_indicators':
                    indicators[indicator_name] = self._create_iv_indicators(data, indicator_config)
                elif indicator_name == 'premium_indicators':
                    indicators[indicator_name] = self._create_premium_indicators(data, indicator_config)
                elif indicator_name == 'trending_oi_pa':
                    indicators[indicator_name] = self._create_trending_oi_pa(data, indicator_config)
                elif indicator_name == 'greek_sentiment':
                    indicators[indicator_name] = self._create_greek_sentiment(data, indicator_config)
                elif indicator_name == 'vwap_indicators':
                    indicators[indicator_name] = self._create_vwap_indicators(data, indicator_config)
                elif indicator_name == 'volume_indicators':
                    indicators[indicator_name] = self._create_volume_indicators(data, indicator_config)
                else:
                    self.logger.warning(f"Unknown indicator type: {indicator_name}")
        
        return indicators
    
    def _filter_by_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Filter data for a specific timeframe."""
        # This is a placeholder implementation
        # In a real system, this would filter the data based on the timeframe
        # For now, we'll just return the original data
        return data
    
    def _filter_by_dte(self, data: pd.DataFrame, dte_range: List[int]) -> pd.DataFrame:
        """Filter data for a specific DTE range."""
        if 'DTE' not in data.columns:
            self.logger.warning("DTE column not found in data")
            return data
            
        if len(dte_range) != 2:
            self.logger.warning(f"Invalid DTE range: {dte_range}")
            return data
            
        return data[(data['DTE'] >= dte_range[0]) & (data['DTE'] <= dte_range[1])]
    
    def _parse_list_param(self, param, default_str, convert_func=None):
        """
        Parse a parameter that could be a list or a comma-separated string.
        
        Args:
            param: The parameter to parse
            default_str: Default string value if param is None
            convert_func: Function to convert each item (e.g., int, float)
            
        Returns:
            List of parsed values
        """
        if param is None:
            param = default_str
            
        if isinstance(param, list):
            result = param
        elif isinstance(param, str):
            result = [item.strip() for item in param.split(',')]
        else:
            # Try to convert to string and split
            try:
                result = [item.strip() for item in str(param).split(',')]
            except:
                self.logger.warning(f"Could not parse parameter: {param}, using default: {default_str}")
                result = [item.strip() for item in default_str.split(',')]
        
        # Apply conversion function if provided
        if convert_func is not None:
            try:
                result = [convert_func(item) for item in result]
            except:
                self.logger.warning(f"Could not convert items in {result}, using as is")
                
        return result
    
    def _create_ema_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create EMA indicators based on configuration."""
        # Create an empty dictionary to store all columns, will convert to DataFrame at the end
        # This avoids DataFrame fragmentation warnings
        result_dict = {}
        
        # Parse configuration parameters
        periods = self._parse_list_param(config.get('periods'), "20,100,200", int)
        timeframes = self._parse_list_param(config.get('timeframes'), "5m,10m,15m")
        price_columns = self._parse_list_param(config.get('price_columns'), "ATM_Straddle_Premium,ATM_CE_Premium,ATM_PE_Premium")
        
        use_slope = config.get('use_slope', True)
        use_crossover = config.get('use_crossover', True)
        use_alignment = config.get('use_alignment', True)
        
        for timeframe in timeframes:
            # Filter data for this timeframe if needed
            timeframe_data = self._filter_by_timeframe(data, timeframe)
            
            for price_col in price_columns:
                if price_col not in timeframe_data.columns:
                    self.logger.warning(f"Price column {price_col} not found in data")
                    continue
                
                # Calculate EMAs for each period
                for period in periods:
                    ema_col = f"{price_col}_{timeframe}_EMA{period}"
                    ema_values = timeframe_data[price_col].ewm(span=period, adjust=False).mean()
                    result_dict[ema_col] = ema_values
                    
                    # Calculate EMA position (price relative to EMA)
                    if price_col in timeframe_data.columns:
                        position_col = f"{price_col}_{timeframe}_EMA{period}_Position"
                        position_values = (timeframe_data[price_col] - ema_values) / ema_values
                        result_dict[position_col] = position_values
                
                # Calculate EMA slope if enabled
                if use_slope:
                    for period in periods:
                        ema_col = f"{price_col}_{timeframe}_EMA{period}"
                        slope_col = f"{price_col}_{timeframe}_EMA{period}_Slope"
                        if ema_col in result_dict:
                            slope_values = pd.Series(result_dict[ema_col]).pct_change(5)
                            result_dict[slope_col] = slope_values
                
                # Calculate EMA crossovers if enabled
                if use_crossover and len(periods) >= 2:
                    periods_sorted = sorted(periods)
                    for i in range(len(periods_sorted) - 1):
                        short_period = periods_sorted[i]
                        long_period = periods_sorted[i + 1]
                        short_ema = f"{price_col}_{timeframe}_EMA{short_period}"
                        long_ema = f"{price_col}_{timeframe}_EMA{long_period}"
                        crossover_col = f"{price_col}_{timeframe}_EMA{short_period}_{long_period}_Crossover"
                        
                        # Initialize crossover values
                        crossover_values = pd.Series(0, index=data.index)
                        
                        # Calculate crossovers row by row to avoid Series comparison issues
                        for i, idx in enumerate(data.index):
                            # Skip first row as we need a previous value
                            if i == 0:
                                continue
                                
                            # Get previous index safely
                            prev_idx = data.index[i-1]
                            
                            # Get current and previous values as scalars
                            try:
                                if short_ema in result_dict and long_ema in result_dict:
                                    curr_short = float(result_dict[short_ema].iloc[i])
                                    curr_long = float(result_dict[long_ema].iloc[i])
                                    prev_short = float(result_dict[short_ema].iloc[i-1])
                                    prev_long = float(result_dict[long_ema].iloc[i-1])
                                    
                                    # Skip if any value is NaN
                                    if (np.isnan(curr_short) or np.isnan(curr_long) or 
                                        np.isnan(prev_short) or np.isnan(prev_long)):
                                        continue
                                        
                                    # Bullish crossover
                                    if curr_short > curr_long and prev_short <= prev_long:
                                        crossover_values.iloc[i] = 1
                                    # Bearish crossover
                                    elif curr_short < curr_long and prev_short >= prev_long:
                                        crossover_values.iloc[i] = -1
                            except (ValueError, TypeError, IndexError):
                                # Skip if any value can't be converted to float or index error
                                continue
                        
                        result_dict[crossover_col] = crossover_values
                
                # Calculate EMA alignment if enabled
                if use_alignment and len(periods) >= 2:
                    alignment_col = f"{price_col}_{timeframe}_EMA_Alignment"
                    
                    # Initialize alignment values
                    alignment_values = pd.Series(0, index=data.index)
                    
                    # Get EMA column names in order
                    all_emas = [f"{price_col}_{timeframe}_EMA{period}" for period in sorted(periods)]
                    
                    # Calculate alignment for each row individually
                    for i, idx in enumerate(data.index):
                        # Check for bullish alignment (shorter above longer)
                        bullish_aligned = True
                        bearish_aligned = True
                        
                        # Get all EMA values for this row
                        ema_values = []
                        for ema_col in all_emas:
                            if ema_col in result_dict:
                                try:
                                    val = float(result_dict[ema_col].iloc[i])
                                    if np.isnan(val):  # Skip if any value is NaN
                                        bullish_aligned = False
                                        bearish_aligned = False
                                        break
                                    ema_values.append(val)
                                except (ValueError, TypeError, IndexError):
                                    bullish_aligned = False
                                    bearish_aligned = False
                                    break
                        
                        # Skip if we don't have all values
                        if len(ema_values) != len(all_emas):
                            continue
                            
                        # Check bullish alignment (each EMA above the next)
                        for j in range(len(ema_values) - 1):
                            if ema_values[j] <= ema_values[j + 1]:
                                bullish_aligned = False
                                break
                        
                        # Check bearish alignment (each EMA below the next)
                        for j in range(len(ema_values) - 1):
                            if ema_values[j] >= ema_values[j + 1]:
                                bearish_aligned = False
                                break
                        
                        if bullish_aligned:
                            alignment_values.iloc[i] = 1
                        elif bearish_aligned:
                            alignment_values.iloc[i] = -1
                    
                    result_dict[alignment_col] = alignment_values
                
                # Calculate overall trend strength
                trend_strength_col = f"{price_col}_{timeframe}_Trend_Strength"
                
                # Initialize trend strength values
                trend_strength_values = pd.Series(0, index=data.index)
                
                # Combine position, slope, crossover, and alignment signals
                for period in periods:
                    position_col = f"{price_col}_{timeframe}_EMA{period}_Position"
                    if position_col in result_dict:
                        # Apply function to each value individually
                        for i, idx in enumerate(data.index):
                            try:
                                pos_val = float(result_dict[position_col].iloc[i])
                                if not np.isnan(pos_val):  # Check for NaN
                                    if pos_val > 0:
                                        trend_strength_values.iloc[i] += 1
                                    elif pos_val < 0:
                                        trend_strength_values.iloc[i] -= 1
                            except (ValueError, TypeError, IndexError):
                                continue
                
                if use_slope:
                    for period in periods:
                        slope_col = f"{price_col}_{timeframe}_EMA{period}_Slope"
                        if slope_col in result_dict:
                            # Apply function to each value individually
                            for i, idx in enumerate(data.index):
                                try:
                                    slope_val = float(result_dict[slope_col].iloc[i])
                                    if not np.isnan(slope_val):  # Check for NaN
                                        if slope_val > 0:
                                            trend_strength_values.iloc[i] += 1
                                        elif slope_val < 0:
                                            trend_strength_values.iloc[i] -= 1
                                except (ValueError, TypeError, IndexError):
                                    continue
                
                if use_crossover and len(periods) >= 2:
                    for i in range(len(periods) - 1):
                        short_period = periods[i]
                        long_period = periods[i + 1]
                        crossover_col = f"{price_col}_{timeframe}_EMA{short_period}_{long_period}_Crossover"
                        if crossover_col in result_dict:
                            # Add crossover signal to trend strength
                            for i, idx in enumerate(data.index):
                                try:
                                    cross_val = float(result_dict[crossover_col].iloc[i])
                                    if not np.isnan(cross_val):  # Check for NaN
                                        trend_strength_values.iloc[i] += cross_val
                                except (ValueError, TypeError, IndexError):
                                    continue
                
                if use_alignment:
                    alignment_col = f"{price_col}_{timeframe}_EMA_Alignment"
                    if alignment_col in result_dict:
                        # Add alignment signal to trend strength (with double weight)
                        for i, idx in enumerate(data.index):
                            try:
                                align_val = float(result_dict[alignment_col].iloc[i])
                                if not np.isnan(align_val):  # Check for NaN
                                    trend_strength_values.iloc[i] += align_val * 2
                            except (ValueError, TypeError, IndexError):
                                continue
                
                # Normalize trend strength to range [-1, 1]
                max_strength = (len(periods) + (len(periods) if use_slope else 0) + 
                               (len(periods) - 1 if use_crossover else 0) + 
                               (2 if use_alignment else 0))
                
                if max_strength > 0:
                    for i, idx in enumerate(data.index):
                        try:
                            strength_val = float(trend_strength_values.iloc[i])
                            if not np.isnan(strength_val):  # Check for NaN
                                trend_strength_values.iloc[i] = strength_val / max_strength
                        except (ValueError, TypeError, IndexError):
                            continue
                
                result_dict[trend_strength_col] = trend_strength_values
        
        # Convert dictionary to DataFrame
        result = pd.DataFrame(result_dict, index=data.index)
        return result
    
    def _create_atr_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create ATR indicators based on configuration."""
        # Create a simplified implementation that avoids boolean indexing
        result_dict = {}
        
        # Parse configuration parameters
        atr_period = int(config.get('atr_period', 14))
        percentile_lookback = int(config.get('percentile_lookback', 100))
        
        # Parse list parameters
        ema_periods = self._parse_list_param(config.get('ema_periods'), "20,50,100", int)
        timeframes = self._parse_list_param(config.get('timeframes'), "5m,10m,15m")
        
        for timeframe in timeframes:
            # Filter data for this timeframe if needed
            timeframe_data = self._filter_by_timeframe(data, timeframe)
            
            # Check if required columns exist
            required_cols = ['High', 'Low', 'Close']
            if not all(col in timeframe_data.columns for col in required_cols):
                self.logger.warning(f"Required columns {required_cols} not found in data for ATR calculation")
                continue
            
            # Calculate True Range
            tr_col = f"TR_{timeframe}"
            tr_values = np.maximum(
                timeframe_data['High'] - timeframe_data['Low'],
                np.maximum(
                    np.abs(timeframe_data['High'] - timeframe_data['Close'].shift(1)),
                    np.abs(timeframe_data['Low'] - timeframe_data['Close'].shift(1))
                )
            )
            result_dict[tr_col] = tr_values
            
            # Calculate ATR
            atr_col = f"ATR_{timeframe}"
            atr_values = tr_values.rolling(window=atr_period).mean()
            result_dict[atr_col] = atr_values
            
            # Calculate ATR percentile
            percentile_col = f"ATR_{timeframe}_Percentile"
            percentile_values = pd.Series(np.nan, index=data.index)
            
            # Calculate percentile for each row based on historical data
            for i, idx in enumerate(data.index):
                if i > percentile_lookback:  # Need enough history
                    try:
                        historical_data = atr_values.iloc[:i+1]
                        if len(historical_data) > 1:
                            lookback = min(percentile_lookback, len(historical_data) - 1)
                            historical_subset = historical_data.iloc[-lookback:]
                            current_value = historical_data.iloc[-1]
                            if not pd.isna(current_value):
                                percentile = (historical_subset < current_value).mean()
                                percentile_values.iloc[i] = percentile
                    except (ValueError, TypeError, IndexError):
                        continue
            
            result_dict[percentile_col] = percentile_values
            
            # Calculate ATR/Close ratio
            if 'Close' in timeframe_data.columns:
                ratio_col = f"ATR_{timeframe}_Ratio"
                ratio_values = pd.Series(np.nan, index=data.index)
                
                for i, idx in enumerate(data.index):
                    try:
                        atr_val = float(atr_values.iloc[i])
                        close_val = float(timeframe_data['Close'].iloc[i])
                        if not np.isnan(atr_val) and not np.isnan(close_val) and close_val != 0:
                            ratio_values.iloc[i] = atr_val / close_val
                    except (ValueError, TypeError, IndexError):
                        continue
                
                result_dict[ratio_col] = ratio_values
                
                # Calculate ATR/Close ratio percentile
                ratio_percentile_col = f"ATR_{timeframe}_Ratio_Percentile"
                ratio_percentile_values = pd.Series(np.nan, index=data.index)
                
                # Calculate percentile for each row based on historical data
                for i, idx in enumerate(data.index):
                    if i > percentile_lookback:  # Need enough history
                        try:
                            historical_data = ratio_values.iloc[:i+1]
                            if len(historical_data) > 1:
                                lookback = min(percentile_lookback, len(historical_data) - 1)
                                historical_subset = historical_data.iloc[-lookback:]
                                current_value = historical_data.iloc[-1]
                                if not pd.isna(current_value):
                                    percentile = (historical_subset < current_value).mean()
                                    ratio_percentile_values.iloc[i] = percentile
                        except (ValueError, TypeError, IndexError):
                            continue
                
                result_dict[ratio_percentile_col] = ratio_percentile_values
            
            # Calculate ATR EMAs
            for period in ema_periods:
                atr_ema_col = f"ATR_{timeframe}_EMA{period}"
                atr_ema_values = atr_values.ewm(span=period, adjust=False).mean()
                result_dict[atr_ema_col] = atr_ema_values
            
            # Calculate ATR expansion/contraction
            expansion_col = f"ATR_{timeframe}_Expansion"
            expansion_values = pd.Series(0, index=data.index)
            
            # ATR is expanding if current ATR > ATR EMA
            if ema_periods:
                first_ema_col = f"ATR_{timeframe}_EMA{ema_periods[0]}"
                for i, idx in enumerate(data.index):
                    try:
                        atr_val = float(atr_values.iloc[i])
                        ema_val = float(result_dict[first_ema_col].iloc[i])
                        if not np.isnan(atr_val) and not np.isnan(ema_val):
                            if atr_val > ema_val:
                                expansion_values.iloc[i] = 1
                            elif atr_val < ema_val:
                                expansion_values.iloc[i] = -1
                    except (ValueError, TypeError, IndexError):
                        continue
            
            result_dict[expansion_col] = expansion_values
            
            # Determine volatility regime based on ATR percentile
            regime_col = f"Volatility_Regime_{timeframe}"
            regime_values = pd.Series("Normal_Vol", index=data.index)  # Default
            
            for i, idx in enumerate(data.index):
                try:
                    pct_val = float(percentile_values.iloc[i])
                    if not np.isnan(pct_val):
                        if pct_val < 0.2:
                            regime_values.iloc[i] = "Very_Low_Vol"
                        elif pct_val >= 0.2 and pct_val < 0.4:
                            regime_values.iloc[i] = "Low_Vol"
                        elif pct_val >= 0.4 and pct_val < 0.6:
                            regime_values.iloc[i] = "Normal_Vol"
                        elif pct_val >= 0.6 and pct_val < 0.8:
                            regime_values.iloc[i] = "High_Vol"
                        elif pct_val >= 0.8:
                            regime_values.iloc[i] = "Extreme_Vol"
                except (ValueError, TypeError, IndexError):
                    continue
            
            result_dict[regime_col] = regime_values
        
        # Convert dictionary to DataFrame
        result = pd.DataFrame(result_dict, index=data.index)
        return result
    
    def _create_iv_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create IV indicators based on configuration."""
        # Simplified implementation that avoids boolean indexing
        result_dict = {}
        
        # Parse configuration parameters
        lookback_period = int(config.get('lookback_period', 60))
        dte_specific_lookback = config.get('dte_specific_lookback', True)
        
        # Parse DTE buckets
        dte_buckets = self._parse_list_param(config.get('dte_buckets'), "0,7,14,30,60,90", int)
        
        use_skew = config.get('use_skew', True)
        use_term_structure = config.get('use_term_structure', True)
        
        # Check if required columns exist
        if 'IV' not in data.columns:
            self.logger.warning("IV column not found in data for IV indicators calculation")
            return pd.DataFrame(index=data.index)
        
        # Initialize IV regime
        iv_regime_values = pd.Series("Normal_IV", index=data.index)  # Default
        
        if dte_specific_lookback and 'DTE' in data.columns:
            # Process each DTE bucket separately
            for i in range(len(dte_buckets) - 1):
                dte_min = dte_buckets[i]
                dte_max = dte_buckets[i + 1]
                dte_key = f"{dte_min}-{dte_max}"
                
                # Get lookback period for this DTE range
                bucket_lookback = lookback_period
                if f"dte_lookback_mapping.{dte_key}" in config:
                    bucket_lookback = int(config.get(f"dte_lookback_mapping.{dte_key}", lookback_period))
                
                # Filter data for this DTE range
                dte_data = self._filter_by_dte(data, [dte_min, dte_max])
                
                # Calculate IV percentile for this DTE range
                iv_percentile_col = f"IV_Percentile_DTE_{dte_min}_{dte_max}"
                iv_percentile_values = pd.Series(np.nan, index=data.index)
                
                if not dte_data.empty:
                    # Calculate percentile for each row based on historical data
                    for idx in dte_data.index:
                        if idx in data.index:  # Make sure index exists in original data
                            i = data.index.get_loc(idx)
                            historical_data = dte_data.loc[:idx, 'IV']
                            if len(historical_data) > 1:  # Need at least some history
                                lookback = min(bucket_lookback, len(historical_data) - 1)
                                historical_subset = historical_data.iloc[-lookback:]
                                try:
                                    current_value = float(historical_data.iloc[-1])
                                    if not np.isnan(current_value):
                                        percentile = (historical_subset < current_value).mean()
                                        iv_percentile_values.iloc[i] = percentile
                                except (ValueError, TypeError, IndexError):
                                    continue
                
                result_dict[iv_percentile_col] = iv_percentile_values
        else:
            # Calculate overall IV percentile
            iv_percentile_col = 'IV_Percentile'
            iv_percentile_values = pd.Series(np.nan, index=data.index)
            
            # Calculate percentile for each row based on historical data
            for i, idx in enumerate(data.index):
                if i > lookback_period:  # Need enough history
                    try:
                        historical_data = data.loc[:idx, 'IV']
                        if len(historical_data) > 1:
                            lookback = min(lookback_period, len(historical_data) - 1)
                            historical_subset = historical_data.iloc[-lookback:]
                            current_value = float(historical_data.iloc[-1])
                            if not np.isnan(current_value):
                                percentile = (historical_subset < current_value).mean()
                                iv_percentile_values.iloc[i] = percentile
                    except (ValueError, TypeError, IndexError):
                        continue
            
            result_dict[iv_percentile_col] = iv_percentile_values
        
        # Use DTE-specific percentiles if available
        iv_percentile_cols = [col for col in result_dict.keys() if 'IV_Percentile' in col]
        
        if iv_percentile_cols:
            # Calculate average percentile across all DTE buckets
            iv_percentile_avg_values = pd.Series(np.nan, index=data.index)
            
            for i, idx in enumerate(data.index):
                percentiles = []
                for col in iv_percentile_cols:
                    try:
                        pct_val = float(result_dict[col].iloc[i])
                        if not np.isnan(pct_val):
                            percentiles.append(pct_val)
                    except (ValueError, TypeError, IndexError):
                        continue
                
                if percentiles:
                    iv_percentile_avg_values.iloc[i] = sum(percentiles) / len(percentiles)
            
            result_dict['IV_Percentile_Avg'] = iv_percentile_avg_values
            
            # Determine regime based on average percentile
            for i, idx in enumerate(data.index):
                try:
                    avg_pct = float(iv_percentile_avg_values.iloc[i])
                    if not np.isnan(avg_pct):
                        if avg_pct < 0.2:
                            iv_regime_values.iloc[i] = "Very_Low_IV"
                        elif avg_pct >= 0.2 and avg_pct < 0.4:
                            iv_regime_values.iloc[i] = "Low_IV"
                        elif avg_pct >= 0.4 and avg_pct < 0.6:
                            iv_regime_values.iloc[i] = "Normal_IV"
                        elif avg_pct >= 0.6 and avg_pct < 0.8:
                            iv_regime_values.iloc[i] = "High_IV"
                        elif avg_pct >= 0.8:
                            iv_regime_values.iloc[i] = "Extreme_IV"
                except (ValueError, TypeError, IndexError):
                    continue
        elif 'IV_Percentile' in result_dict:
            # Use overall percentile if DTE-specific not available
            for i, idx in enumerate(data.index):
                try:
                    pct_val = float(iv_percentile_values.iloc[i])
                    if not np.isnan(pct_val):
                        if pct_val < 0.2:
                            iv_regime_values.iloc[i] = "Very_Low_IV"
                        elif pct_val >= 0.2 and pct_val < 0.4:
                            iv_regime_values.iloc[i] = "Low_IV"
                        elif pct_val >= 0.4 and pct_val < 0.6:
                            iv_regime_values.iloc[i] = "Normal_IV"
                        elif pct_val >= 0.6 and pct_val < 0.8:
                            iv_regime_values.iloc[i] = "High_IV"
                        elif pct_val >= 0.8:
                            iv_regime_values.iloc[i] = "Extreme_IV"
                except (ValueError, TypeError, IndexError):
                    continue
        
        result_dict['IV_Regime'] = iv_regime_values
        
        # Calculate IV skew if enabled
        if use_skew and 'Call_IV' in data.columns and 'Put_IV' in data.columns:
            iv_skew_values = pd.Series(np.nan, index=data.index)
            
            for i, idx in enumerate(data.index):
                try:
                    call_iv = float(data.loc[idx, 'Call_IV'])
                    put_iv = float(data.loc[idx, 'Put_IV'])
                    if not np.isnan(call_iv) and not np.isnan(put_iv) and call_iv != 0:
                        iv_skew_values.iloc[i] = put_iv / call_iv
                except (ValueError, TypeError, IndexError):
                    continue
            
            result_dict['IV_Skew'] = iv_skew_values
            
            # Calculate IV skew percentile
            iv_skew_percentile_values = pd.Series(np.nan, index=data.index)
            
            # Calculate percentile for each row based on historical data
            for i, idx in enumerate(data.index):
                if i > lookback_period:  # Need enough history
                    try:
                        historical_data = iv_skew_values.iloc[:i+1]
                        if len(historical_data) > 1:
                            lookback = min(lookback_period, len(historical_data) - 1)
                            historical_subset = historical_data.iloc[-lookback:]
                            current_value = historical_data.iloc[-1]
                            if not pd.isna(current_value):
                                percentile = (historical_subset < current_value).mean()
                                iv_skew_percentile_values.iloc[i] = percentile
                    except (ValueError, TypeError, IndexError):
                        continue
            
            result_dict['IV_Skew_Percentile'] = iv_skew_percentile_values
            
            # Determine IV skew regime
            iv_skew_regime_values = pd.Series("Normal_Skew", index=data.index)  # Default
            
            for i, idx in enumerate(data.index):
                try:
                    pct_val = float(iv_skew_percentile_values.iloc[i])
                    if not np.isnan(pct_val):
                        if pct_val < 0.2:
                            iv_skew_regime_values.iloc[i] = "Very_Low_Skew"
                        elif pct_val >= 0.2 and pct_val < 0.4:
                            iv_skew_regime_values.iloc[i] = "Low_Skew"
                        elif pct_val >= 0.4 and pct_val < 0.6:
                            iv_skew_regime_values.iloc[i] = "Normal_Skew"
                        elif pct_val >= 0.6 and pct_val < 0.8:
                            iv_skew_regime_values.iloc[i] = "High_Skew"
                        elif pct_val >= 0.8:
                            iv_skew_regime_values.iloc[i] = "Extreme_Skew"
                except (ValueError, TypeError, IndexError):
                    continue
            
            result_dict['IV_Skew_Regime'] = iv_skew_regime_values
        
        # Convert dictionary to DataFrame
        result = pd.DataFrame(result_dict, index=data.index)
        return result
    
    def _create_premium_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create premium indicators based on configuration."""
        # Simplified implementation that avoids boolean indexing
        # Return empty DataFrame for now
        return pd.DataFrame(index=data.index)
    
    def _create_trending_oi_pa(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create trending OI with PA indicators based on configuration."""
        # Simplified implementation that avoids boolean indexing
        # Return empty DataFrame for now
        return pd.DataFrame(index=data.index)
    
    def _create_greek_sentiment(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create Greek sentiment indicators based on configuration."""
        # Simplified implementation that avoids boolean indexing
        # Return empty DataFrame for now
        return pd.DataFrame(index=data.index)
    
    def _create_vwap_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create VWAP indicators based on configuration."""
        # Simplified implementation that avoids boolean indexing
        # Return empty DataFrame for now
        return pd.DataFrame(index=data.index)
    
    def _create_volume_indicators(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create volume indicators based on configuration."""
        # Simplified implementation that avoids boolean indexing
        # Return empty DataFrame for now
        return pd.DataFrame(index=data.index)
