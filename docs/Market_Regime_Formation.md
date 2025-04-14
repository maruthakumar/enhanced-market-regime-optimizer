# Market Regime Formation Process

The Market Regime Formation Process is the core component of the Enhanced Market Regime Optimizer, combining multiple analyses to identify 18 distinct market regimes that guide trading strategy optimization.

## Overview

The Market Regime Formation Process integrates Greek sentiment analysis, trending OI with PA analysis, IV skew and percentile analysis, and technical indicators to classify the current market state into one of 18 distinct regimes. This classification provides a framework for optimizing trading strategies based on the specific characteristics of each regime.

## The 18 Market Regimes

The system forms 18 market regimes by combining:

### Directional Component (7 states)
- Strong_Bullish
- Mild_Bullish
- Sideways_To_Bullish
- Neutral/Sideways
- Sideways_To_Bearish
- Mild_Bearish
- Strong_Bearish

### Volatility Component (3 states)
- High_Volatile
- Normal_Volatile
- Low_Volatile

### Transitional Regimes (3 additional states)
- Bullish_To_Bearish_Transition
- Bearish_To_Bullish_Transition
- Volatility_Expansion

## Methodology

The Market Regime Formation Process follows these steps:

1. **Calculate Directional Component**:
   - Integrate Greek sentiment analysis (40% weight by default)
   - Integrate trending OI with PA analysis (30% weight by default)
   - Integrate IV skew analysis (10% weight by default)
   - Integrate EMA and VWAP indicators (10% and 5% weight by default)
   - Produce a directional score between -1 (strongly bearish) and 1 (strongly bullish)
   - Classify into one of 7 directional states based on thresholds

2. **Calculate Volatility Component**:
   - Integrate ATR relative to price (5% weight by default)
   - Integrate IV percentile (5% weight by default)
   - Produce a volatility score between -1 (low volatility) and 1 (high volatility)
   - Classify into one of 3 volatility states based on thresholds

3. **Detect Transitions**:
   - Analyze changes in directional and volatility components
   - Identify potential regime transitions
   - Calculate transition probability scores

4. **Classify Market Regime**:
   - Check for transitional regimes first (if transition probability > 0.5)
   - Combine directional and volatility states to determine regime
   - Calculate confidence score based on component confidences

5. **Calculate Rolling Market Regime**:
   - Apply 5-minute rolling window to smooth regime assignments
   - Calculate regime confidence scores
   - Provide stable regime assignments for trading strategy optimization

6. **Dynamic Weight Adjustment**:
   - Optimize component weights based on historical performance
   - Implement rolling weight optimization, adaptive threshold adjustment, and performance-based weight adjustment
   - Continuously improve regime classification accuracy

## DTE-Specific Implementation

The Market Regime Formation Process includes DTE-specific implementation:

- **Weekly Expiry Focus (0-6 DTE)** for Nifty & Sensex
- **Full Range DTE Analysis** for BankNifty & MidcapNifty (monthly expiry)

This allows for more precise market regime identification based on the specific expiry cycle.

## Time Series Storage

The implementation includes CSV storage for time series market regime data:

- 1-minute rolling market regime data
- Daily aggregated market regime data
- Standardized format for easy consolidation with strategies

## Visualization

The Market Regime Formation Process includes comprehensive visualizations:

- Daily regime calendars and transition heatmaps
- Regime distribution charts and timelines
- Component contribution analysis
- Intraday pattern visualization
- Strike-specific OI pattern visualization
- Interactive dashboards

## Implementation

The Market Regime Formation Process is implemented in the `market_regime_classifier.py` module in the `src/` directory. The implementation includes:

- Functions for calculating directional and volatility components
- Transition detection algorithms
- Market regime classification logic
- Rolling market regime calculation
- Dynamic weight adjustment
- CSV storage and visualization utilities

## Usage

```python
from src.market_regime_classifier import MarketRegimeClassifier

# Initialize classifier with custom configuration
config = {
    'greek_sentiment_weight': 0.4,
    'trending_oi_pa_weight': 0.3,
    'iv_skew_weight': 0.1,
    'ema_weight': 0.1,
    'vwap_weight': 0.05,
    'atr_weight': 0.05,
    'use_dynamic_weights': True,
    'window_size': 30
}
classifier = MarketRegimeClassifier(config)

# Classify market regime
result = classifier.classify_regime(
    data_frame,
    greek_sentiment_column='Greek_Sentiment',
    trending_oi_pa_column='OI_PA_Regime',
    iv_skew_column='IV_Skew',
    ema_column='EMA_Signal',
    vwap_column='VWAP_Signal',
    atr_column='ATR',
    specific_dte=5  # For DTE-specific analysis
)

# Save to CSV
classifier.save_to_csv(result, 'market_regimes.csv')

# Visualize regimes
classifier.visualize_regimes(result, 'visualizations/')
```
