# Minute-by-Minute Market Regime Classifier

## Overview

This package provides a memory-optimized minute-by-minute market regime classifier for NSE Nifty 50 options trading. The classifier processes market data in small chunks to identify market regimes with confidence scores at the minute level, enabling more precise entry and exit points and strategy selection for intraday trading.

## Features

- **Minute-level Regime Classification**: Identifies market regimes at minute-level granularity
- **Multiple Indicator Integration**:
  - Trending OI with PA analysis at 5-minute timeframe
  - EMA indicators at 15, 10, 5, and 3-minute timeframes
  - VWAP indicators with previous day's VWAP reference
  - Greek sentiment analysis with minute-level aggregation
- **Dynamic Weightage System**:
  - Time-of-day specific weightage adjustments
  - Volatility-based weight scaling
  - Performance tracking for each indicator
- **Memory Optimization**:
  - Chunk-based processing with configurable chunk size
  - Aggressive memory cleanup after processing each component
  - Memory usage monitoring with automatic garbage collection
- **Comprehensive Error Handling**:
  - Graceful degradation when individual indicators fail
  - Detailed logging for troubleshooting
- **Strategy Integration**:
  - Regime-specific strategy recommendations
  - Confidence scores for each classification

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn tqdm psutil
   ```
3. Extract the package to your desired location

## Directory Structure

```
minute_regime_classifier/
├── src/
│   ├── minute_regime_classifier.py       # Main classifier implementation
│   └── optimal_indicator_weightages.py   # Weightage analysis script
├── utils/
│   ├── feature_engineering/
│   │   ├── base/                         # Base indicator module
│   │   ├── trending_oi_pa/               # Trending OI with PA analysis
│   │   ├── ema_indicators/               # EMA indicators at multiple timeframes
│   │   ├── vwap_indicators/              # VWAP indicators
│   │   └── greek_sentiment/              # Greek sentiment analysis
│   └── dynamic_weight_adjustment/        # Dynamic weightage system
├── docs/
│   └── comprehensive_report.md           # Detailed analysis report
└── examples/
    └── run_minute_regime_classifier.py   # Example usage script
```

## Quick Start

1. Import the necessary modules:
   ```python
   from src.minute_regime_classifier import MinuteRegimeClassifier, process_date_range
   ```

2. Process data for a specific date range:
   ```python
   start_date = '2025-01-01'
   end_date = '2025-04-15'
   results = process_date_range(start_date, end_date)
   ```

3. Check the results:
   ```python
   print(f"Results saved to {results.get('output_dir')}")
   print(f"Combined results file: {results.get('combined_file')}")
   print(f"Report file: {results.get('report_file')}")
   ```

For a complete example, see `examples/run_minute_regime_classifier.py`.

## Optimal Indicator Weightages

Based on our analysis, we recommend the following weightages for minute-by-minute market regime classification:

- Trending OI with PA: 30.0%
- Greek Sentiment: 25.0%
- EMA Indicators: 20.0%
- VWAP Indicators: 20.0%
- Other Indicators: 5.0%

For more detailed recommendations, including time-of-day and volatility-based adjustments, see the comprehensive report in `docs/comprehensive_report.md`.

## Implementation Recommendations

1. **Use Dynamic Weightages**:
   - Start with the overall optimal weightages as default
   - Adjust weightages based on time of day
   - Further adjust based on current market volatility
   - Implement a feedback mechanism to fine-tune weightages based on recent performance

2. **Confidence Score Thresholds**:
   - Use a minimum confidence threshold of 0.7 for trading decisions
   - Consider higher thresholds (0.8+) for larger position sizes
   - Monitor regime persistence for additional confirmation

3. **Strategy Integration**:
   - Each regime has associated optimal strategies as identified in the classifier
   - Use the recommended strategies for each identified regime
   - Consider regime transitions as potential entry/exit signals

## Customization

You can customize the classifier by modifying the configuration parameters in `src/minute_regime_classifier.py`. Key parameters include:

- `chunk_size`: Size of data chunks for processing (default: 1000)
- `memory_threshold_mb`: Memory threshold for garbage collection (default: 1000)
- Indicator weights and parameters in the `_get_default_config()` method

## Troubleshooting

- **Memory Issues**: Reduce the `chunk_size` parameter if you encounter memory errors
- **Missing Columns**: The classifier includes fallback mechanisms for missing columns, but ensure your data includes at least:
  - datetime or date/time columns
  - underlying_price or close price
  - volume (optional, will use dummy values if missing)
  - option greeks (delta, gamma, theta, vega) for Greek sentiment analysis

## License

This package is provided for your personal use only. All rights reserved.

## Contact

For support or questions, please contact the developer.
