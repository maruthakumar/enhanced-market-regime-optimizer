# Enhanced Market Regime Optimizer

The Enhanced Market Regime Optimizer is a comprehensive framework for identifying market regimes based on Greek sentiment analysis, trending OI with PA, IV skew and percentile analysis, and technical indicators. It provides a robust foundation for optimizing trading strategies based on market conditions.

## Overview

The Enhanced Market Regime Optimizer identifies 18 distinct market regimes by combining:

1. **Directional Component** (from Greek sentiment and OI analysis)
2. **Volatility Component** (from technical indicators and IV analysis)
3. **Transitional Regimes** (based on changes in components)

These market regimes provide a framework for optimizing trading strategies based on specific market conditions.

## Key Components

### 1. Greek Sentiment Analysis

The Greek sentiment analysis component tracks changes in options Greeks (Delta, Vega, Theta) to determine market sentiment:

- Uses aggregate opening values for Delta, Vega, and Theta
- Calculates minute-to-minute changes
- Implements dynamic weight adjustment
- Classifies sentiment into directional categories with confidence metrics
- Supports sideways and transitional market states

### 2. Trending OI with PA Analysis

The trending OI with PA analysis component provides comprehensive analysis of Open Interest patterns from an option seller's perspective:

- Analyzes 15 strikes (ATM plus 7 above and 7 below)
- Implements OI pattern recognition (long build-up, short build-up, etc.)
- Includes OI velocity and acceleration calculation
- Provides combined call-put pattern identification
- Incorporates historical pattern behavior analysis
- Detects pattern divergence across multiple strikes
- Tracks pattern performance over time

### 3. IV Skew and Percentile Analysis

The IV skew and percentile analysis component examines the distribution of Implied Volatility:

- Calculates IV percentile for specific DTEs
- Implements IV skew analysis across strikes
- Provides term structure analysis across expirations
- Includes ATM straddle IV tracking
- Classifies IV skew into market sentiment categories with confidence metrics

### 4. Technical Indicators

The technical indicators component calculates various technical indicators:

- EMA indicators (20, 50, 200-period)
- VWAP indicators
- ATR indicators
- RSI and other oscillators

## Market Regimes

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

## Features

- **DTE-Specific Analysis**: Tailored analysis for weekly and monthly expiries
- **Dynamic Weight Adjustment**: Continuously optimizes component weights based on historical performance and component divergence
- **Confidence-Weighted Signals**: Incorporates signal confidence in regime classification
- **Component Divergence Analysis**: Detects and accounts for conflicting signals between components
- **Historical Pattern Analysis**: Tracks and learns from historical pattern performance
- **Rolling Market Regime Calculation**: Provides stable regime assignments with 1-minute granularity
- **CSV Storage**: Stores time series market regime data for easy integration with strategies
- **Comprehensive Visualizations**: Includes regime calendars, transition heatmaps, and component contribution analysis

## Installation

```bash
git clone https://github.com/maruthakumar/enhanced-market-regime-optimizer.git
cd enhanced-market-regime-optimizer
pip install -r requirements.txt
```

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
    iv_skew_column='IV_Skew_Classification',
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

## Documentation

For detailed documentation on each component, see the `docs` directory:

- `Greek_sentiment.md`: Documentation for Greek sentiment analysis
- `Trending_OI_PA.md`: Documentation for trending OI with PA analysis
- `IV_Skew_Percentile.md`: Documentation for IV skew and percentile analysis
- `Market_Regime_Formation.md`: Documentation for the market regime formation process
- `Component_Divergence.md`: Documentation for component divergence analysis
- `Historical_Pattern_Analysis.md`: Documentation for historical pattern behavior analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped improve this project
- Special thanks to the options trading community for their insights and feedback
