# Unified Market Regime Pipeline

## Overview

The Unified Market Regime Pipeline is the central orchestration component of the Enhanced Market Regime Optimizer system. It serves as the main entry point for the entire pipeline, coordinating the flow of data through various processing stages from market data input to strategy optimization and results visualization.

This pipeline integrates all components of the market regime identification system, connecting with the existing consolidator and optimizer components to provide a comprehensive solution for market regime-based trading strategy optimization.

## Key Features

- Complete market regime identification pipeline
- Integration with consolidator for strategy data processing
- Connection to dimensional optimizer for strategy optimization
- Multi-timeframe analysis for more robust regime identification
- Time-of-day adjustments to account for intraday market dynamics
- Comprehensive logging and error handling
- Visualization capabilities for results analysis

## Pipeline Components

The Unified Market Regime Pipeline consists of the following main components:

### 1. Feature Engineering Components

These components calculate various indicators from market data:

- **Greek Sentiment Analysis**: Analyzes options Greek values to determine market sentiment
- **Trending OI with PA Analysis**: Analyzes open interest and price action trends
- **IV Skew Analysis**: Analyzes implied volatility skew patterns
- **EMA Indicators**: Calculates exponential moving averages and related signals
- **VWAP Indicators**: Calculates volume-weighted average price and related signals

### 2. Market Regime Classification

This component combines the outputs from feature engineering components to classify the market into one of 18 predefined regimes.

### 3. Multi-timeframe Analysis

This component integrates market regime classifications from multiple timeframes for more robust regime identification.

### 4. Time-of-day Adjustments

This component adjusts the weights of different indicators based on the time of day to account for intraday market dynamics.

### 5. Results Processing

This component saves the results to CSV files and generates visualizations.

### 6. Consolidator Integration

This component prepares data for the consolidator, which combines strategy data with market regime data.

## Configuration Options

The Unified Market Regime Pipeline offers extensive configuration options to customize its behavior. These options can be specified in a configuration dictionary passed to the `MarketRegimePipeline` constructor.

### General Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | 'output' | Directory where output files will be saved |
| `use_multi_timeframe` | boolean | True | Whether to use multi-timeframe analysis |
| `use_time_of_day_adjustments` | boolean | True | Whether to use time-of-day adjustments |

### Component Weights

These weights determine the contribution of each component to the final market regime classification:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `component_weights.greek_sentiment` | float | 0.20 | Weight for Greek sentiment analysis |
| `component_weights.trending_oi_pa` | float | 0.30 | Weight for trending OI with PA analysis |
| `component_weights.iv_skew` | float | 0.20 | Weight for IV skew analysis |
| `component_weights.ema_indicators` | float | 0.15 | Weight for EMA indicators |
| `component_weights.vwap_indicators` | float | 0.15 | Weight for VWAP indicators |

### Multi-timeframe Configuration

These parameters configure the multi-timeframe analysis:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeframe_weights.5m` | float | 0.20 | Weight for 5-minute timeframe |
| `timeframe_weights.15m` | float | 0.30 | Weight for 15-minute timeframe |
| `timeframe_weights.1h` | float | 0.30 | Weight for 1-hour timeframe |
| `timeframe_weights.1d` | float | 0.20 | Weight for 1-day timeframe |

### Time-of-day Adjustments

These parameters configure the time-of-day adjustments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_of_day_weights.opening` | float | 1.2 | Weight multiplier for opening (9:15-9:45) |
| `time_of_day_weights.morning` | float | 1.0 | Weight multiplier for morning (9:45-12:00) |
| `time_of_day_weights.lunch` | float | 0.8 | Weight multiplier for lunch (12:00-13:00) |
| `time_of_day_weights.afternoon` | float | 1.0 | Weight multiplier for afternoon (13:00-14:30) |
| `time_of_day_weights.closing` | float | 1.2 | Weight multiplier for closing (14:30-15:30) |

### Component-specific Configuration

Each feature engineering component has its own configuration options:

#### Greek Sentiment Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `greek_sentiment_config.vega_weight` | float | 0.5 | Weight for Vega in Greek sentiment calculation |
| `greek_sentiment_config.delta_weight` | float | 0.4 | Weight for Delta in Greek sentiment calculation |
| `greek_sentiment_config.theta_weight` | float | 0.1 | Weight for Theta in Greek sentiment calculation |
| `greek_sentiment_config.use_dynamic_weights` | boolean | True | Whether to use dynamic weights for Greeks |

#### Trending OI with PA Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trending_oi_pa_config.strikes_above_atm` | int | 7 | Number of strikes above ATM to analyze |
| `trending_oi_pa_config.strikes_below_atm` | int | 7 | Number of strikes below ATM to analyze |
| `trending_oi_pa_config.use_velocity` | boolean | True | Whether to use OI velocity in analysis |
| `trending_oi_pa_config.use_acceleration` | boolean | True | Whether to use OI acceleration in analysis |

#### IV Skew Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iv_skew_config.skew_threshold` | float | 0.1 | Threshold for significant skew |
| `iv_skew_config.use_percentile` | boolean | True | Whether to use percentile-based thresholds |

#### EMA Indicators Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ema_indicators_config.periods` | list | [20, 50, 200] | EMA periods to calculate |
| `ema_indicators_config.use_crossovers` | boolean | True | Whether to use EMA crossovers in analysis |
| `ema_indicators_config.instrument_weights.underlying` | float | 0.1 | Weight for underlying price in EMA analysis |
| `ema_indicators_config.instrument_weights.straddle` | float | 0.4 | Weight for ATM straddle in EMA analysis |
| `ema_indicators_config.instrument_weights.ce` | float | 0.25 | Weight for ATM CE in EMA analysis |
| `ema_indicators_config.instrument_weights.pe` | float | 0.25 | Weight for ATM PE in EMA analysis |

#### VWAP Indicators Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vwap_indicators_config.use_bands` | boolean | True | Whether to use VWAP bands in analysis |
| `vwap_indicators_config.band_multiplier` | float | 2.0 | Multiplier for VWAP bands |
| `vwap_indicators_config.instrument_weights.underlying` | float | 0.1 | Weight for underlying price in VWAP analysis |
| `vwap_indicators_config.instrument_weights.straddle` | float | 0.4 | Weight for ATM straddle in VWAP analysis |
| `vwap_indicators_config.instrument_weights.ce` | float | 0.25 | Weight for ATM CE in VWAP analysis |
| `vwap_indicators_config.instrument_weights.pe` | float | 0.25 | Weight for ATM PE in VWAP analysis |

## Market Regime Indicators Selection

The Unified Market Regime Pipeline uses a combination of indicators to classify the market regime. You can customize which indicators to use and their weights through the configuration options.

### Recommended Indicator Combinations

Different market conditions may benefit from different indicator combinations:

#### For Trending Markets

- Trending OI with PA: 0.35
- EMA Indicators: 0.25
- Greek Sentiment: 0.20
- VWAP Indicators: 0.10
- IV Skew: 0.10

#### For Volatile Markets

- IV Skew: 0.30
- Greek Sentiment: 0.25
- Trending OI with PA: 0.20
- EMA Indicators: 0.15
- VWAP Indicators: 0.10

#### For Sideways Markets

- VWAP Indicators: 0.30
- Trending OI with PA: 0.25
- IV Skew: 0.20
- EMA Indicators: 0.15
- Greek Sentiment: 0.10

### Indicator Selection Guidelines

When selecting indicators and their weights, consider the following guidelines:

1. **Market Characteristics**: Different markets may respond better to different indicators. For example, equity indices may respond better to EMA and VWAP indicators, while options markets may respond better to Greek sentiment and IV skew.

2. **Timeframe**: Shorter timeframes may benefit from more weight on faster-moving indicators like VWAP, while longer timeframes may benefit from more weight on trend-following indicators like EMA.

3. **Volatility**: During high volatility periods, consider increasing the weight of IV skew and Greek sentiment indicators.

4. **Liquidity**: In highly liquid markets, price-based indicators like EMA and VWAP may be more reliable, while in less liquid markets, options-based indicators like Greek sentiment and IV skew may provide better signals.

## Dimensional Optimization Configuration

The Unified Market Regime Pipeline prepares data for dimensional optimization, which identifies the most relevant dimensions for strategy optimization. The following dimensions are available:

### DTE (Days to Expiry)

DTE is a critical dimension for options strategies. Different strategies may perform better at different points in the expiry cycle.

Configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension_selection.dte_ranges` | list | [[0, 1], [2, 3], [4, 7], [8, 14], [15, 30]] | DTE ranges for optimization |
| `dimension_selection.use_dte` | boolean | True | Whether to use DTE as a dimension |

### Zone

Trading zones represent different time periods during the trading day. Different strategies may perform better in different zones.

Configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension_selection.zones` | list | ['Opening', 'Morning', 'Lunch', 'Afternoon', 'Closing'] | Trading zones for optimization |
| `dimension_selection.use_zone` | boolean | True | Whether to use Zone as a dimension |

### Market Regime

Market regimes represent different market conditions. Different strategies may perform better in different market regimes.

Configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension_selection.use_market_regime` | boolean | True | Whether to use Market Regime as a dimension |
| `dimension_selection.regime_groups` | dict | See below | Grouping of market regimes for optimization |

Default regime groups:

```python
{
    'Bullish': ['Strong_Bullish', 'Bullish', 'Moderately_Bullish'],
    'Mildly_Bullish': ['Weakly_Bullish', 'Bullish_Consolidation', 'Sideways_To_Bullish'],
    'Neutral': ['Neutral_Bullish_Bias', 'Neutral', 'Neutral_Bearish_Bias'],
    'Mildly_Bearish': ['Weakly_Bearish', 'Bearish_Consolidation', 'Sideways_To_Bearish'],
    'Bearish': ['Strong_Bearish', 'Bearish', 'Moderately_Bearish'],
    'Volatile': ['Bullish_Volatile', 'Bearish_Volatile']
}
```

### Day

Different days of the week may exhibit different market behavior. Different strategies may perform better on different days.

Configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension_selection.use_day` | boolean | True | Whether to use Day as a dimension |
| `dimension_selection.day_groups` | dict | See below | Grouping of days for optimization |

Default day groups:

```python
{
    'Start_Week': ['Monday', 'Tuesday'],
    'Mid_Week': ['Wednesday'],
    'End_Week': ['Thursday', 'Friday']
}
```

## Market Regime Confidence Level

The Unified Market Regime Pipeline calculates a confidence score for each market regime classification. This confidence score represents the system's certainty in the classification and can be used to filter out low-confidence signals.

### Confidence Score Calculation

The confidence score is calculated based on several factors:

1. **Component Agreement**: The degree to which different components agree on the market regime.
2. **Component Confidence**: The confidence of each individual component in its signal.
3. **Historical Performance**: The historical accuracy of the market regime classification in similar conditions.

The confidence score ranges from 0 to 1, with higher values indicating higher confidence.

### Using Confidence Scores

Confidence scores can be used in several ways:

1. **Filtering**: Only consider market regime classifications with confidence scores above a certain threshold.
2. **Weighting**: Weight strategy decisions based on the confidence score.
3. **Risk Management**: Adjust position sizes based on the confidence score.

Configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_threshold` | float | 0.6 | Minimum confidence score for a valid market regime classification |
| `use_confidence_weighting` | boolean | True | Whether to weight strategy decisions based on confidence scores |

## Step-by-Step Usage Guide

### Basic Usage

```python
from unified_market_regime_pipeline import MarketRegimePipeline

# Initialize the pipeline with default configuration
pipeline = MarketRegimePipeline()

# Run the pipeline on a single data file
result = pipeline.run_pipeline(
    data_file='data/market_data/nifty_options_data.csv',
    output_file='output/nifty_market_regimes.csv',
    dte=5,
    timeframe='5m'
)

# Print the results
print(f"Processed {len(result)} rows")
print(f"Market regimes: {result['market_regime'].value_counts()}")
```

### Multi-timeframe Analysis

```python
# Run the pipeline with multi-timeframe analysis
result = pipeline.run_multi_timeframe_pipeline(
    data_files={
        '5m': 'data/market_data/nifty_options_data_5m.csv',
        '15m': 'data/market_data/nifty_options_data_15m.csv',
        '1h': 'data/market_data/nifty_options_data_1h.csv',
        '1d': 'data/market_data/nifty_options_data_1d.csv'
    },
    output_dir='output/multi_timeframe',
    dte=5
)
```

### Custom Configuration

```python
# Create a custom configuration
config = {
    'output_dir': 'output/custom',
    'component_weights': {
        'greek_sentiment': 0.25,
        'trending_oi_pa': 0.25,
        'iv_skew': 0.20,
        'ema_indicators': 0.15,
        'vwap_indicators': 0.15
    },
    'use_multi_timeframe': True,
    'timeframe_weights': {
        '5m': 0.25,
        '15m': 0.30,
        '1h': 0.30,
        '1d': 0.15
    },
    'use_time_of_day_adjustments': True,
    'time_of_day_weights': {
        'opening': 1.3,
        'morning': 1.0,
        'lunch': 0.7,
        'afternoon': 1.0,
        'closing': 1.3
    },
    'greek_sentiment_config': {
        'vega_weight': 0.6,
        'delta_weight': 0.3,
        'theta_weight': 0.1,
        'use_dynamic_weights': True
    },
    'trending_oi_pa_config': {
        'strikes_above_atm': 5,
        'strikes_below_atm': 5,
        'use_velocity': True,
        'use_acceleration': True
    },
    'iv_skew_config': {
        'skew_threshold': 0.15,
        'use_percentile': True
    },
    'ema_indicators_config': {
        'periods': [20, 50, 100],
        'use_crossovers': True,
        'instrument_weights': {
            'underlying': 0.05,
            'straddle': 0.45,
            'ce': 0.25,
            'pe': 0.25
        }
    },
    'vwap_indicators_config': {
        'use_bands': True,
        'band_multiplier': 2.5,
        'instrument_weights': {
            'underlying': 0.05,
            'straddle': 0.45,
            'ce': 0.25,
            'pe': 0.25
        }
    },
    'confidence_threshold': 0.7,
    'use_confidence_weighting': True
}

# Initialize the pipeline with custom configuration
pipeline = MarketRegimePipeline(config)

# Run the pipeline
result = pipeline.run_pipeline(
    data_file='data/market_data/nifty_options_data.csv',
    output_file='output/custom/nifty_market_regimes.csv',
    dte=5,
    timeframe='5m'
)
```

### Preparing Data for Consolidator

```python
# Process data and prepare for consolidator
result = pipeline.process_data(
    data=data,
    dte=5,
    timeframe='5m'
)

# Prepare data for consolidator
consolidator_data = pipeline.prepare_for_consolidator(
    data=result,
    dte=5
)

# Save data for consolidator
consolidator_data.to_csv('output/consolidator_input.csv', index=False)
```

## Integration with Consolidator and Optimizer

The Unified Market Regime Pipeline integrates with the consolidator and optimizer components of the Enhanced Market Regime Optimizer system.

### Consolidator Integration

The pipeline prepares data for the consolidator using the `prepare_for_consolidator` method. This method formats the market regime data in a way that can be consumed by the consolidator.

The consolidator then combines the market regime data with strategy data to create a consolidated dataset that includes both market regimes and strategy performance.

### Optimizer Integration

The consolidated data from the consolidator is used by the optimizer to find the optimal parameters for each strategy based on the selected dimensions (DTE, Zone, market regime, day).

The optimizer uses the market regime classifications and confidence scores to determine which strategies perform best in which market regimes.

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Missing or incomplete market data

**Symptoms**: The pipeline fails to process data or produces incomplete results.

**Solutions**:
- Check that the input data file exists and contains all required columns.
- Ensure that the data has the correct format and column names.
- Check for missing values in the data and handle them appropriately.

#### Issue: Low confidence scores

**Symptoms**: The pipeline produces market regime classifications with low confidence scores.

**Solutions**:
- Check that the input data is of good quality and contains all necessary information.
- Adjust the component weights to give more weight to more reliable indicators.
- Consider using multi-timeframe analysis to improve confidence.

#### Issue: Inconsistent market regime classifications

**Symptoms**: The pipeline produces different market regime classifications for similar market conditions.

**Solutions**:
- Use multi-timeframe analysis to improve consistency.
- Adjust the component weights to give more weight to more stable indicators.
- Consider using a longer lookback period for historical pattern analysis.

#### Issue: Performance issues

**Symptoms**: The pipeline takes too long to process data or uses too much memory.

**Solutions**:
- Reduce the number of strikes analyzed in the trending OI with PA analysis.
- Use a smaller subset of data for testing.
- Optimize the code for performance, especially in the feature engineering components.

### Logging and Debugging

The Unified Market Regime Pipeline includes comprehensive logging to help diagnose issues. By default, logs are written to the `market_regime_pipeline.log` file.

To enable more detailed logging, you can adjust the logging level:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_regime_pipeline_debug.log"),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

### Optimizing for Speed

To optimize the pipeline for speed, consider the following:

1. **Reduce the number of strikes analyzed**: The trending OI with PA analysis can be computationally expensive if analyzing many strikes. Consider reducing the number of strikes above and below ATM.

2. **Use selective feature engineering**: Only calculate the indicators that are most relevant for your specific use case.

3. **Optimize data loading**: Load data efficiently, possibly using a database or optimized file formats like Parquet.

4. **Parallelize processing**: Consider parallelizing the processing of different timeframes or different symbols.

### Optimizing for Accuracy

To optimize the pipeline for accuracy, consider the following:

1. **Use multi-timeframe analysis**: Combining signals from multiple timeframes can improve accuracy.

2. **Adjust component weights**: Give more weight to indicators that perform better in your specific market.

3. **Use dynamic weight adjustment**: Allow the system to adjust weights based on historical performance.

4. **Incorporate more data sources**: Consider adding additional data sources or indicators to improve accuracy.

### Memory Management

The pipeline can consume significant memory when processing large datasets. To manage memory usage:

1. **Process data in chunks**: Instead of loading all data at once, process it in smaller chunks.

2. **Clean up temporary data**: Remove temporary data structures when they are no longer needed.

3. **Use efficient data structures**: Use memory-efficient data structures like NumPy arrays instead of Python lists where appropriate.

## Conclusion

The Unified Market Regime Pipeline is a powerful tool for identifying market regimes and optimizing trading strategies based on market conditions. By properly configuring and using the pipeline, you can gain valuable insights into market behavior and improve your trading performance.

For more information on specific components of the pipeline, refer to the following documentation:

- [Market Regime Formation](Market_Regime_Formation.md)
- [Consolidation](Consolidation.md)
- [Dimension Selection](Dimension_Selection.md)
- [Results Visualization](Results_Visualization.md)
- [PostgreSQL Integration](PostgreSQL_Integration.md)
- [GDFL Live Data Feed](GDFL_Live_Data_Feed.md)