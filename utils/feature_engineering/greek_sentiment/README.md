# Greek Sentiment Analysis for Market Regime Optimization

This package contains specialized modules for analyzing market regimes based on Greek sentiment indicators (Delta, Vega, Theta) from options data. These modules are designed to be integrated with the enhanced-market-regime-optimizer codebase.

## Overview

The Greek sentiment analysis approach provides valuable insights into market dynamics by analyzing how options Greeks change over time. This implementation includes:

1. Dynamic weight adjustment based on historical performance
2. Hierarchical market regime classification (5 regimes)
3. Optimized data processing for large datasets
4. 1-minute regime detection for intraday trading
5. Comprehensive visualization tools

## Modules

- `greek_dynamic_weight_adjuster.py`: Implements dynamic weight adjustment based on historical performance
- `greek_regime_classifier.py`: Enhanced market regime classification with multiple methods
- `greek_data_processor.py`: Optimized data processing for large datasets
- `greek_market_regime.py`: Core functionality for Greek sentiment-based market regimes
- `greek_daily_market_regime_analysis.py`: Day trading analysis script
- `greek_test_implementation.py`: Testing framework for verification

## Installation

1. Copy all files to your project directory:
   ```
   cp greek_*.py /path/to/your/project/
   ```

2. Import the modules in your code:
   ```python
   from greek_data_processor import process_data_efficiently
   from greek_dynamic_weight_adjuster import GreekDynamicWeightAdjuster
   from greek_regime_classifier import GreekRegimeClassifier
   from greek_market_regime import process_greek_sentiment
   ```

## Usage for 1-Minute Data

For 1-minute data analysis, use the following approach:

```python
# In your main trading code
from greek_data_processor import process_data_efficiently
from greek_market_regime import process_greek_sentiment, classify_greek_market_regimes

# Process new 1-minute data
def process_new_minute_data(new_data, config):
    # Process Greek sentiment
    sentiment_data = process_greek_sentiment(new_data, config)
    
    # Classify regime
    regime_data = classify_greek_market_regimes(sentiment_data, config)
    
    # Get latest regime
    current_regime = regime_data['Market_Regime'].iloc[-1]
    
    # Check for regime transition
    regime_shift = regime_data['Regime_Shift'].iloc[-1] if 'Regime_Shift' in regime_data.columns else False
    
    # Generate trading signal
    if regime_shift:
        # Handle regime transition
        print(f"REGIME TRANSITION: {current_regime}")
        # Execute your trading logic based on new regime
    
    return current_regime
```

## Configuration

For optimal performance with 1-minute data:

```python
# In your configuration
config.set('data_processing', 'chunk_size', '1000')  # Smaller chunks for faster processing
config.set('market_regime', 'use_clustering', 'false')  # Use faster threshold-based classification
config.set('market_regime', 'regime_lookback_window', '30')  # 30-minute lookback
```

## Daily Analysis

For day traders, run the daily analysis script:

```bash
python greek_daily_market_regime_analysis.py --intraday --time-granularity 1
```

## Integration with Existing Codebase

These modules are designed to be used alongside other market regime models. The "greek_" prefix clearly distinguishes these Greek sentiment-based modules from other approaches.

## Testing

Run the test implementation script to verify functionality:

```bash
python greek_test_implementation.py
```

This will generate sample data, process it through the system, and create visualizations to demonstrate the functionality.
