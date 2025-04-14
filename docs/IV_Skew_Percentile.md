# IV Skew and Percentile Analysis

This document describes the IV Skew and Percentile Analysis implemented in the Enhanced Market Regime Optimizer, which examines the distribution of Implied Volatility across strikes and expirations to identify market sentiment.

## Overview

The IV Skew and Percentile Analysis component examines the relationship between call and put implied volatilities across different strikes and calculates IV percentiles to identify market sentiment and volatility expectations. This provides valuable insights into market fear, complacency, and potential directional moves.

## Key Concepts

### 1. IV Skew

IV skew measures the difference in implied volatility between OTM puts and OTM calls:

- **Positive Skew (Put Skew)**: Higher IV for OTM puts compared to OTM calls, indicating market fear of downside moves
- **Negative Skew (Call Skew)**: Higher IV for OTM calls compared to OTM calls, indicating expectations of upside moves
- **Neutral Skew**: Similar IV levels for OTM puts and calls, indicating balanced expectations

### 2. IV Percentile

IV percentile places current IV levels in historical context:

- **High Percentile (>80%)**: Current IV is high relative to historical levels, indicating elevated fear or uncertainty
- **Medium Percentile (20-80%)**: Current IV is within normal historical range
- **Low Percentile (<20%)**: Current IV is low relative to historical levels, indicating complacency

### 3. ATM Straddle

ATM straddle premium and IV provide insights into expected magnitude of price movement:

- **High ATM IV**: Market expects large price movements (direction-neutral)
- **Low ATM IV**: Market expects small price movements
- **Changes in ATM IV**: Indicate shifting volatility expectations

### 4. Term Structure

IV term structure examines the relationship between IVs across different expirations:

- **Normal Term Structure**: Higher IV for longer-dated options
- **Inverted Term Structure**: Higher IV for shorter-dated options, indicating near-term uncertainty
- **Flat Term Structure**: Similar IV across expirations

## Implementation

### IV Skew Calculation

The system calculates IV skew as follows:

1. Identify ATM strike (closest to current price)
2. Find 25-delta OTM put and 25-delta OTM call (approximately 5% OTM)
3. Calculate IV skew as: OTM Put IV - OTM Call IV
4. Normalize skew relative to ATM IV for comparison across time

### IV Skew Classification

IV skew is classified into market sentiment categories:

- **Strong Bearish**: IV skew > 0.05 (high put skew)
- **Mild Bearish**: IV skew between 0.02 and 0.05
- **Sideways to Bearish**: IV skew between 0.01 and 0.02
- **Neutral**: IV skew between -0.01 and 0.01
- **Sideways to Bullish**: IV skew between -0.02 and -0.01
- **Mild Bullish**: IV skew between -0.05 and -0.02
- **Strong Bullish**: IV skew < -0.05 (high call skew)

### IV Percentile Calculation

The system calculates IV percentile as follows:

1. Maintain a rolling window of historical IV data (default: 60 periods)
2. Calculate the percentile rank of current IV within this window
3. Optionally calculate DTE-specific percentiles for more accurate comparison

### Confidence Calculation

The system calculates confidence in IV skew signals:

1. Higher confidence for extreme IV skew values (>0.1 or <-0.1)
2. Higher confidence for extreme IV percentiles (>80% or <20%)
3. Lower confidence for middle-range values
4. Adjust confidence based on IV skew magnitude and consistency

## Advanced Features

### DTE-Specific Analysis

The system performs expiration-specific analysis:

1. Calculate separate IV metrics for weekly and monthly expirations
2. Compare IV skew patterns across different DTEs
3. Identify term structure anomalies

### IV Surface Analysis

The system analyzes the entire IV surface:

1. Calculate skew slope (rate of IV change across strikes)
2. Identify volatility smiles and smirks
3. Detect unusual IV patterns across the surface

### Historical Pattern Analysis

The system tracks IV skew patterns over time:

1. Identify historical instances of similar IV skew patterns
2. Calculate subsequent market performance
3. Determine predictive value of specific IV skew patterns

## Usage in Market Regime Classification

The IV Skew component contributes to market regime classification:

1. Provides directional component (bullish/bearish) based on IV skew classification
2. Contributes to volatility component based on IV percentile
3. Includes confidence score based on skew magnitude and percentile extremes
4. Default weight in market regime classification: 10%
5. Weight is dynamically adjusted based on historical accuracy

## Example

For a scenario with high put skew (0.06) and high IV percentile (85%):

1. IV skew is classified as Strong_Bearish
2. High IV percentile indicates elevated market fear
3. Confidence score is high (0.9) due to extreme values
4. This contributes to a bearish directional component and high volatility component in market regime classification

## Conclusion

IV Skew and Percentile Analysis provides valuable insights into market sentiment and volatility expectations. By examining the relationship between call and put implied volatilities and placing current IV levels in historical context, this component enhances the accuracy of market regime classification in the Enhanced Market Regime Optimizer.
