# Greek Sentiment Analysis

Greek sentiment analysis is a key component of the Enhanced Market Regime Optimizer, providing the directional component for market regime classification.

## Overview

Greek sentiment analysis tracks changes in options Greeks (Delta, Vega, Theta) to determine market sentiment. Unlike traditional technical analysis that focuses on price and volume, Greek sentiment analysis captures the underlying options market positioning, which often leads price action.

## Methodology

The Greek sentiment analysis process follows these steps:

1. **Record Opening Values**:
   - Capture aggregate opening values for Delta, Vega, and Theta at the start of the trading day
   - These serve as baseline reference points for all subsequent calculations

2. **Calculate Minute-to-Minute Changes**:
   - For each minute, calculate the difference between current Greek values and opening values
   - Track these changes separately for calls and puts

3. **Aggregate Changes for Calls vs. Puts**:
   - Sum the changes across all relevant strikes
   - Create separate aggregates for calls and puts

4. **Calculate Weighted Sentiment Score**:
   - Combine Delta, Vega, and Theta components with appropriate weights:
   - `Sentiment_Score = (Delta_Weight × Delta_Component) + (Vega_Weight × Vega_Component) + (Theta_Weight × Theta_Component)`
   - Default weights: Vega (50%), Delta (40%), Theta (10%)
   - Weights are dynamically adjusted based on historical performance

5. **Classify Sentiment**:
   - Strong_Bullish: Sentiment score > 0.5
   - Mild_Bullish: Sentiment score between 0.2 and 0.5
   - Sideways_To_Bullish: Sentiment score between 0.1 and 0.2
   - Neutral/Sideways: Sentiment score between -0.1 and 0.1
   - Sideways_To_Bearish: Sentiment score between -0.2 and -0.1
   - Mild_Bearish: Sentiment score between -0.5 and -0.2
   - Strong_Bearish: Sentiment score < -0.5

## Interpretation of Greek Changes

The sentiment classification is based on the following interpretations:

- **Delta Changes**:
  - Increasing call delta → bullish
  - Decreasing call delta → bearish
  - Increasing put delta → bearish
  - Decreasing put delta → bullish

- **Vega Changes**:
  - Increasing call vega → call options being bought → bullish
  - Decreasing call vega → call options being sold → bearish
  - Increasing put vega → put options being bought → bearish
  - Decreasing put vega → put options being sold → bullish

- **Theta Changes**:
  - Increasing call theta → call options being sold → bearish
  - Decreasing call theta → call options being bought → bullish
  - Increasing put theta → put options being sold → bullish
  - Decreasing put theta → put options being bought → bearish

## Dynamic Weight Adjustment

The Greek sentiment analysis includes dynamic weight adjustment to optimize the contribution of each Greek:

1. **Rolling Weight Optimization**:
   - Uses a sliding window approach to continuously optimize weights based on recent historical data
   - Implements an objective function that maximizes F1 score to balance precision and recall
   - Ensures weights sum to 1 while allowing individual weights to vary between 0 and 1

2. **Adaptive Threshold Adjustment**:
   - Dynamically modifies sentiment classification thresholds based on market volatility
   - Uses percentiles of historical sentiment scores to determine appropriate thresholds
   - Automatically adjusts during high volatility periods when standard thresholds may be less effective

3. **Performance-Based Weight Adjustment**:
   - Tracks the predictive accuracy of each component (Delta, Vega, Theta) individually
   - Implements a reinforcement learning approach that increases weights for components with higher predictive accuracy
   - Provides immediate feedback to adjust weights based on prediction results

## Integration with Market Regime Formation

Greek sentiment analysis provides the directional component for market regime formation, which is combined with the volatility component to determine the final market regime. The sentiment classification is used to identify the directional state (Strong_Bullish, Mild_Bullish, Neutral/Sideways, etc.), which is then combined with the volatility state (High_Volatile, Normal_Volatile, Low_Volatile) to form the 18 market regimes.

## Implementation

The Greek sentiment analysis is implemented in the `greek_sentiment_analysis.py` module in the `utils/feature_engineering/greek_sentiment/` directory. The implementation includes:

- Functions for calculating Greek changes
- Sentiment score calculation with dynamic weights
- Sentiment classification with adaptive thresholds
- Integration with the market regime classifier

## Usage

```python
from utils.feature_engineering.greek_sentiment import GreekSentimentAnalyzer

# Initialize analyzer
analyzer = GreekSentimentAnalyzer()

# Analyze Greek sentiment
sentiment = analyzer.analyze(options_data)

# Get sentiment classification
classification = analyzer.classify_sentiment(sentiment)
```
