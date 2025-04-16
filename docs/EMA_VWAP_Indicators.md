# EMA and VWAP Indicators Analysis

EMA and VWAP indicators analysis is an important component of the Enhanced Market Regime Optimizer, providing additional directional signals for market regime classification with special emphasis on ATM option instruments.

## Overview

The Enhanced Market Regime Optimizer uses Exponential Moving Averages (EMA) and Volume Weighted Average Price (VWAP) indicators to identify trend direction and potential support/resistance levels. Unlike traditional implementations that focus solely on the underlying price, our enhanced approach calculates these indicators on both the underlying price AND ATM option instruments (straddle, CE, and PE), with dynamic weightage adjustment that prioritizes option instruments over the underlying.

## Methodology

### EMA Indicators Analysis

The EMA indicators analysis process follows these steps:

1. **Calculate EMAs for Multiple Instruments**:
   - Calculate EMAs with periods 20, 50, and 200 for the underlying price
   - Calculate EMAs with the same periods for ATM straddle price
   - Calculate EMAs with the same periods for ATM CE price
   - Calculate EMAs with the same periods for ATM PE price

2. **Analyze Price Position Relative to EMAs**:
   - For each instrument, determine if the current price is above or below each EMA
   - Calculate the distance between price and EMAs as a percentage
   - Identify EMA crossovers (e.g., 20 EMA crossing above/below 50 EMA)

3. **Apply Dynamic Weightage**:
   - Assign higher weights to ATM straddle, CE, and PE signals compared to underlying price signals
   - Default weights: ATM straddle (40%), ATM CE (25%), ATM PE (25%), underlying (10%)
   - Dynamically adjust these weights based on market conditions using `utils/dynamic_weight_adjustment/dynamic_weight_adjustment.py`

4. **Calculate EMA Signal Score**:
   - Combine the weighted signals from all instruments
   - Produce an EMA signal score between -1 (strongly bearish) and 1 (strongly bullish)

5. **Classify EMA Signal**:
   - Strong_Bullish: EMA score > 0.5
   - Mild_Bullish: EMA score between 0.2 and 0.5
   - Sideways_To_Bullish: EMA score between 0.1 and 0.2
   - Neutral/Sideways: EMA score between -0.1 and 0.1
   - Sideways_To_Bearish: EMA score between -0.2 and -0.1
   - Mild_Bearish: EMA score between -0.5 and -0.2
   - Strong_Bearish: EMA score < -0.5

### VWAP Indicators Analysis

The VWAP indicators analysis process follows these steps:

1. **Calculate VWAPs for Multiple Instruments**:
   - Calculate VWAP and VWAP bands for the underlying price
   - Calculate VWAP and VWAP bands for ATM straddle price
   - Calculate VWAP and VWAP bands for ATM CE price
   - Calculate VWAP and VWAP bands for ATM PE price

2. **Analyze Price Position Relative to VWAP**:
   - For each instrument, determine if the current price is above or below VWAP
   - Calculate the distance between price and VWAP as a percentage
   - Identify if price is outside VWAP bands (potential overbought/oversold conditions)

3. **Apply Dynamic Weightage**:
   - Assign higher weights to ATM straddle, CE, and PE signals compared to underlying price signals
   - Default weights: ATM straddle (40%), ATM CE (25%), ATM PE (25%), underlying (10%)
   - Dynamically adjust these weights based on market conditions using `utils/dynamic_weight_adjustment/dynamic_weight_adjustment.py`

4. **Calculate VWAP Signal Score**:
   - Combine the weighted signals from all instruments
   - Produce a VWAP signal score between -1 (strongly bearish) and 1 (strongly bullish)

5. **Classify VWAP Signal**:
   - Strong_Bullish: VWAP score > 0.5
   - Mild_Bullish: VWAP score between 0.2 and 0.5
   - Sideways_To_Bullish: VWAP score between 0.1 and 0.2
   - Neutral/Sideways: VWAP score between -0.1 and 0.1
   - Sideways_To_Bearish: VWAP score between -0.2 and -0.1
   - Mild_Bearish: VWAP score between -0.5 and -0.2
   - Strong_Bearish: VWAP score < -0.5

## Timeframe Analysis

Both EMA and VWAP indicators are calculated on multiple timeframes to capture trends at different scales:

- 15-minute timeframe: For longer-term trend identification
- 10-minute timeframe: For medium-term trend identification
- 5-minute timeframe: For short-term trend identification
- 3-minute timeframe: For very short-term trend identification

The signals from different timeframes are combined with appropriate weights, with more weight given to longer timeframes for stability.

## Signal Integration

The signals from ATM straddle, CE, and PE are prioritized over underlying price signals. For example:

- If the underlying price is above its EMA but the ATM straddle is below its EMA, the system will lean toward a bearish interpretation.
- If the ATM CE is above its VWAP but the ATM PE is significantly below its VWAP, this might indicate a potential reversal.

This prioritization is based on the understanding that option instruments often lead price action and provide more nuanced insights into market sentiment.

## Dynamic Weight Adjustment

The dynamic weight adjustment process for EMA and VWAP indicators includes:

1. **Market Volatility-Based Adjustment**:
   - During high volatility periods, increase the weight of ATM straddle signals
   - During low volatility periods, increase the weight of underlying price signals

2. **Time-of-Day Adjustment**:
   - Market opening (first hour): Higher weight to ATM CE and PE signals to capture directional bias
   - Mid-day: Balanced weights across all instruments
   - Market closing (last hour): Higher weight to ATM straddle signals to capture end-of-day sentiment

3. **Performance-Based Adjustment**:
   - Track the predictive accuracy of each instrument's signals
   - Implement a reinforcement learning approach that increases weights for instruments with higher predictive accuracy
   - Provide immediate feedback to adjust weights based on prediction results

## Implementation Details

### EMA Calculation

The EMA calculation for each instrument follows the standard formula:

```python
def calculate_ema(data, period):
    """
    Calculate EMA for the given data and period.
    
    Args:
        data (pd.Series): Price data
        period (int): EMA period
        
    Returns:
        pd.Series: EMA values
    """
    return data.ewm(span=period, adjust=False).mean()
```

### VWAP Calculation

The VWAP calculation for each instrument follows the standard formula:

```python
def calculate_vwap(data, price_column, volume_column):
    """
    Calculate VWAP for the given data.
    
    Args:
        data (pd.DataFrame): Input data
        price_column (str): Column name for price
        volume_column (str): Column name for volume
        
    Returns:
        pd.Series: VWAP values
    """
    typical_price = data[price_column]
    volume = data[volume_column]
    
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    
    return cumulative_tp_vol / cumulative_vol
```

### Signal Score Calculation

The signal score calculation combines signals from all instruments with appropriate weights:

```python
def calculate_signal_score(underlying_signal, straddle_signal, ce_signal, pe_signal, weights):
    """
    Calculate signal score based on weighted signals.
    
    Args:
        underlying_signal (float): Signal from underlying price
        straddle_signal (float): Signal from ATM straddle
        ce_signal (float): Signal from ATM CE
        pe_signal (float): Signal from ATM PE
        weights (dict): Weights for each instrument
        
    Returns:
        float: Signal score
    """
    score = (
        underlying_signal * weights['underlying'] +
        straddle_signal * weights['straddle'] +
        ce_signal * weights['ce'] +
        pe_signal * weights['pe']
    )
    
    return score
```

### Dynamic Weight Adjustment

The dynamic weight adjustment is implemented in the `dynamic_weight_adjustment.py` module:

```python
def adjust_weights(weights, market_volatility, time_of_day, performance_metrics):
    """
    Adjust weights based on market conditions.
    
    Args:
        weights (dict): Current weights
        market_volatility (float): Current market volatility
        time_of_day (str): Current time of day (opening, mid-day, closing)
        performance_metrics (dict): Performance metrics for each instrument
        
    Returns:
        dict: Adjusted weights
    """
    adjusted_weights = weights.copy()
    
    # Volatility-based adjustment
    if market_volatility > 0.5:  # High volatility
        adjusted_weights['straddle'] += 0.1
        adjusted_weights['underlying'] -= 0.1
    elif market_volatility < 0.2:  # Low volatility
        adjusted_weights['straddle'] -= 0.1
        adjusted_weights['underlying'] += 0.1
    
    # Time-of-day adjustment
    if time_of_day == 'opening':
        adjusted_weights['ce'] += 0.05
        adjusted_weights['pe'] += 0.05
        adjusted_weights['straddle'] -= 0.1
    elif time_of_day == 'closing':
        adjusted_weights['straddle'] += 0.1
        adjusted_weights['ce'] -= 0.05
        adjusted_weights['pe'] -= 0.05
    
    # Performance-based adjustment
    for instrument, metric in performance_metrics.items():
        if metric > 0.7:  # High accuracy
            adjusted_weights[instrument] += 0.05
        elif metric < 0.3:  # Low accuracy
            adjusted_weights[instrument] -= 0.05
    
    # Normalize weights to ensure they sum to 1
    total = sum(adjusted_weights.values())
    for instrument in adjusted_weights:
        adjusted_weights[instrument] /= total
    
    return adjusted_weights
```

## Integration with Market Regime Formation

The EMA and VWAP indicators analysis is integrated with the market regime classifier:

```python
# In market_regime_classifier.py
ema_value = self._convert_ema_to_value(data[ema_column])
vwap_value = self._convert_vwap_to_value(data[vwap_column])

directional_component += ema_value * self.indicator_weights['ema']
directional_component += vwap_value * self.indicator_weights['vwap']
```

## Benefits

1. **Improved Accuracy**: Calculating EMAs and VWAPs on both underlying and option instruments provides a more comprehensive view of market sentiment.

2. **Early Warning**: Option instruments often lead price action, allowing for earlier detection of potential market transitions.

3. **Reduced False Signals**: Dynamic weightage adjustment reduces false signals by prioritizing the most relevant instruments in different market conditions.

4. **Adaptive Learning**: Performance-based weight adjustment allows the system to learn from past market behavior and improve its accuracy over time.

## Conclusion

The enhanced EMA and VWAP indicators analysis is a powerful component of the Enhanced Market Regime Optimizer that provides accurate directional signals by analyzing both underlying price and option instruments. By prioritizing ATM straddle, CE, and PE signals over underlying price signals and implementing dynamic weight adjustment, the system helps traders make better-informed decisions based on a more comprehensive view of market sentiment.