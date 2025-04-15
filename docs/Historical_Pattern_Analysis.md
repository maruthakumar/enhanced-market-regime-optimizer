# Historical Pattern Analysis

Historical pattern analysis is a key feature of the Enhanced Market Regime Optimizer that tracks how patterns perform over time and uses this information to improve the accuracy of market regime identification. This document explains how historical pattern analysis works and how it's implemented in the system.

## Overview

Historical pattern analysis involves:

1. Tracking the occurrence of specific OI patterns
2. Measuring their performance over subsequent periods
3. Calculating success rates and average returns
4. Using this historical data to adjust confidence in current pattern identifications

This approach allows the system to learn from past market behavior and improve its accuracy over time.

## Implementation

The historical pattern analysis is implemented in the `TrendingOIWithPAAnalysis` class through the `_analyze_historical_pattern_behavior` method. This method:

1. Identifies patterns in historical data
2. Tracks their performance over a configurable lookback period
3. Calculates success rates and average returns
4. Stores this information in a pattern history database
5. Uses this history to assign confidence scores to current pattern identifications

## Pattern History Database

The system maintains a pattern history database that stores information about each pattern:

```python
self.pattern_history = {
    'Strong_Bullish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
    'Mild_Bullish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
    'Neutral': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
    'Mild_Bearish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0},
    'Strong_Bearish': {'occurrences': 0, 'success_rate': 0, 'avg_return': 0, 'avg_duration': 0}
}
```

This database is persisted to disk using pickle serialization, allowing the system to maintain its learning across sessions:

```python
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
```

## Pattern Performance Tracking

The system tracks pattern performance by:

1. Identifying when a pattern occurs
2. Recording the underlying price at that time
3. Looking ahead a configurable number of periods
4. Calculating the return over that period
5. Determining if the pattern was successful (positive return for bullish patterns, negative return for bearish patterns)

```python
# Calculate return
pattern_return = (future_price - current_price) / current_price

# Update pattern performance
if current_pattern not in pattern_performance:
    pattern_performance[current_pattern] = {'returns': [], 'timestamps': []}

pattern_performance[current_pattern]['returns'].append(pattern_return)
pattern_performance[current_pattern]['timestamps'].append(current_timestamp)
```

## Success Rate Calculation

The system calculates success rates for each pattern:

```python
success_rate = np.mean([1 if (pattern.endswith('Bullish') and r > 0) or (pattern.endswith('Bearish') and r < 0) else 0 for r in returns])
```

A pattern is considered successful if:
- A bullish pattern is followed by a positive return
- A bearish pattern is followed by a negative return

## Confidence Scoring

The system uses the historical success rate to assign confidence scores to current pattern identifications:

```python
if pattern in self.pattern_history and self.pattern_history[pattern]['occurrences'] >= self.min_pattern_occurrences:
    df.loc[i, 'pattern_confidence'] = self.pattern_history[pattern]['success_rate']
else:
    df.loc[i, 'pattern_confidence'] = 0.5  # Neutral confidence if not enough data
```

This ensures that patterns with a strong historical track record are given higher confidence, while patterns with poor historical performance are treated with appropriate caution.

## Time-Weighted Analysis

The system can be configured to give more weight to recent pattern occurrences:

```python
# Example implementation of time-weighted analysis
recent_weight = 0.7
historical_weight = 0.3

recent_success_rate = calculate_recent_success_rate(pattern)
historical_success_rate = self.pattern_history[pattern]['success_rate']

weighted_success_rate = (recent_weight * recent_success_rate) + (historical_weight * historical_success_rate)
```

This allows the system to adapt to changing market conditions while still benefiting from long-term pattern history.

## Configuration Parameters

The historical pattern analysis can be configured through several parameters:

```python
# Historical pattern analysis parameters
self.history_window = int(self.config.get('history_window', 60))  # 60 periods for historical analysis
self.pattern_performance_lookback = int(self.config.get('pattern_performance_lookback', 5))  # Look 5 periods ahead for performance
self.pattern_history_file = self.config.get('pattern_history_file', 'pattern_history.pkl')
self.min_pattern_occurrences = int(self.config.get('min_pattern_occurrences', 10))  # Minimum occurrences for reliable stats
```

These parameters allow users to customize the historical analysis to their specific needs and market conditions.

## Integration with Market Regime Classification

The historical pattern analysis is integrated with the market regime classifier through the confidence scores:

```python
# In trending_oi_pa_analysis.py
regime = {
    'regime': regime,
    'confidence': confidence,
    'pattern': overall_pattern,
    'divergence_score': divergence_score
}

# In market_regime_classifier.py
trending_oi_pa_value = self._convert_trending_oi_pa_to_value(data[trending_oi_pa_column])
trending_oi_pa_confidence = data['pattern_confidence'] if 'pattern_confidence' in data.columns else 0.5
directional_component += trending_oi_pa_value * self.indicator_weights['trending_oi_pa'] * trending_oi_pa_confidence
```

This ensures that the market regime classification takes into account both the current pattern and its historical reliability.

## Visualization

The system provides visualization tools for historical pattern analysis:

```python
plt.figure(figsize=(12, 6))
data.groupby('datetime')['pattern_confidence'].mean().plot()
plt.title('Pattern Confidence Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Confidence')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pattern_confidence.png'))
```

This allows traders to visually monitor pattern confidence levels and identify potential trading opportunities.

## Benefits of Historical Pattern Analysis

1. **Improved Accuracy**: By learning from past pattern performance, the system can improve the accuracy of its market regime identifications.

2. **Adaptive Learning**: The system adapts to changing market conditions by continuously updating its pattern history.

3. **Confidence Calibration**: The confidence scores provided by the system are based on actual historical performance, making them more reliable.

4. **Pattern Discovery**: The system can discover which patterns are most reliable in specific market conditions.

## Conclusion

Historical pattern analysis is a powerful feature of the Enhanced Market Regime Optimizer that allows the system to learn from past market behavior and improve its accuracy over time. By tracking pattern performance and using this information to adjust confidence levels, the system provides more reliable market regime identifications and helps traders make better-informed decisions.
