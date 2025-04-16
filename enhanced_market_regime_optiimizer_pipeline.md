# Enhanced Market Regime Optimizer Pipeline

## Introduction

The Enhanced Market Regime Optimizer is a comprehensive system for identifying market regimes, consolidating strategy data, and optimizing trading strategies based on market conditions. The pipeline processes market data to identify the current market regime, then uses this information to optimize trading strategies based on dimensions such as DTE (Days to Expiry), Zone, market regime, and day of the week.

The system determines which strategies are most suitable for specific market regimes or DTE values, allowing traders to adapt their approach based on changing market conditions.

## Pipeline Overview

The Enhanced Market Regime Optimizer pipeline consists of the following main steps:

1. **Market Regime Formation**: Analyzes market data to classify the current market state into one of 18 predefined regimes.
2. **Consolidation**: Combines strategy data with market regime classifications.
3. **Dimension Selection**: Identifies the most relevant dimensions (DTE, Zone, market regime, day) for optimization.
4. **Optimization**: Runs various optimization algorithms to find optimal parameters for each strategy based on the selected dimensions.
5. **Results Visualization**: Generates visualizations of the optimization results.
6. **Output Storage**: Saves the optimized parameters and performance metrics to a PostgreSQL database.
7. **Live Data Feed Integration**: Integrates with GDFL data feed for real-time market regime identification.

The main entry point for the pipeline is `unified_market_regime_pipeline.py`, which orchestrates the entire process from data input to results visualization. For detailed information on the unified pipeline, see [Unified Market Regime Pipeline Documentation](docs/Unified_Market_Regime_Pipeline.md).

## Pipeline Components

### 1. Market Regime Formation

The Market Regime Formation module classifies the current market state by analyzing a combination of technical indicators. The process involves:

1. **Data Input**: Receiving market data, including price, volume, open interest, and Greek values.
2. **Indicator Calculation**: Calculating individual indicator scores based on the input data using specific implementations in `utils/feature_engineering`.
3. **Regime Classification**: Combining the indicator scores using `src/market_regime_classifier.py` to classify the market into one of 18 predefined regimes.
4. **Dynamic Adjustment**: Adjusting indicator weights and classification thresholds based on market conditions.
5. **Output**: Providing a market regime classification with a confidence score.

For detailed information on the market regime formation process, see [Market Regime Formation Documentation](docs/Market_Regime_Formation.md).

#### Indicator Contributions

Each indicator contributes to the market regime classification as follows:

1. **Trending OI with PA Analysis**:
   * **Concept**: Measures the relationship between price action and open interest to identify bullish or bearish trends.
   * **Implementation**: `utils/feature_engineering/trending_oi_pa/trending_oi_pa_analysis.py`
   * **Documentation**: See [Trending OI PA Documentation](docs/Trending_OI_PA.md) for detailed information.
   * **Key Implementation Details**:
       * Analyzes ATM plus 7 strikes above and 7 strikes below.
       * Calculates OI velocity and acceleration.
       * **Timing**: Primarily uses a 5-minute timeframe for intraday analysis, with a 15-minute timeframe used for confirmation and end-of-day analysis.
   * **Contribution**: Provides a directional bias based on the strength and direction of the trend.

2. **EMA Indicators**:
   * **Concept**: Uses Exponential Moving Averages to identify the trend direction and potential support/resistance levels.
   * **Implementation**: `utils/feature_engineering/ema_indicators/ema_indicators.py`
   * **Documentation**: See [EMA VWAP Indicators Documentation](docs/EMA_VWAP_Indicators.md) for detailed information.
   * **Key Implementation Details**:
       * Calculates EMAs with periods 20, 50, and 200 on both underlying price data **and** ATM Straddle, CE, and PE values.
       * Analyzes price position relative to EMAs for all instruments.
       * **Dynamic Weightage**: More weightage is given to ATM straddle, ATM CE, and ATM PE signals than to underlying price signals.
       * **Timing**: EMAs are calculated on timeframes of 15-minute, 10-minute, 5-minute, and 3-minute to capture trends at different scales.
       * **Signal Integration**: The signals from ATM straddle, CE, and PE EMAs are prioritized over underlying price EMAs. For example, if the underlying price is above its EMA but the ATM straddle is below its EMA, the system will lean toward a bearish interpretation.
       * **Adaptive Weightage**: The weightage between different instruments is dynamically adjusted based on market conditions using `utils/dynamic_weight_adjustment/dynamic_weight_adjustment.py`.
   * **Contribution**: Provides a directional bias and volatility assessment based on EMA crossovers and price relationships across multiple instruments.

3. **VWAP Indicators**:
   * **Concept**: Calculates the Volume Weighted Average Price to identify the average price at which trading has occurred.
   * **Implementation**: `utils/feature_engineering/vwap_indicators/vwap_indicators.py`
   * **Documentation**: See [EMA VWAP Indicators Documentation](docs/EMA_VWAP_Indicators.md) for detailed information.
   * **Key Implementation Details**:
       * Calculates VWAP and VWAP bands for both underlying price **and** ATM Straddle, CE, and PE values.
       * Analyzes price position relative to VWAP for all instruments.
       * **Dynamic Weightage**: More weightage is given to ATM straddle, ATM CE, and ATM PE signals than to underlying price signals.
       * **Timing**: VWAPs are calculated on timeframes of 15-minute, 10-minute, 5-minute, and 3-minute to capture trends at different scales.
       * **Signal Integration**: The signals from ATM straddle, CE, and PE VWAPs are prioritized over underlying price VWAPs. For example, if the underlying price is above its VWAP but the ATM straddle is below its VWAP, the system will lean toward a bearish interpretation.
       * **Adaptive Weightage**: The weightage between different instruments is dynamically adjusted based on market conditions using `utils/dynamic_weight_adjustment/dynamic_weight_adjustment.py`.
   * **Contribution**: Provides a directional bias and identifies potential overbought/oversold conditions based on price deviations from VWAP across multiple instruments.

4. **Greek Sentiment Analysis**:
   * **Concept**: Measures the sentiment of options traders based on changes in Greek values (Delta, Gamma, Theta, Vega).
   * **Implementation**: `utils/feature_engineering/greek_sentiment/greek_sentiment_analysis.py`
   * **Documentation**: See [Greek Sentiment Documentation](docs/Greek_sentiment.md) for detailed information.
   * **Key Implementation Details**:
       * Tracks aggregate opening values for Vega, Delta, and Theta.
       * Calculates minute-to-minute changes from opening values.
   * **Contribution**: Provides a directional bias based on the aggregate sentiment of options traders.

5. **IV Skew Analysis**:
   * **Concept**: Analyzes the implied volatility skew to identify the relative cost of out-of-the-money options.
   * **Implementation**: `utils/feature_engineering/iv_skew/iv_skew_analysis.py`
   * **Documentation**: See [IV Skew Percentile Documentation](docs/IV_Skew_Percentile.md) for detailed information.
   * **Key Implementation Details**:
       * Calculates ATM straddle premium, ATM CE, and ATM PE values.
       * Analyzes IV skew between OTM calls and puts.
   * **Contribution**: Provides a volatility assessment and directional bias based on the skew.

6. **ATR Indicators**:
   * **Concept**: Measures the average true range to quantify market volatility.
   * **Implementation**: `utils/feature_engineering/atr_indicators/`
   * **Key Implementation Details**:
       * Calculates Average True Range for volatility measurement.
   * **Contribution**: Provides a volatility assessment used to classify the market regime.

#### Market Regime Classification

For detailed information on the entire market regime formation process, see [Market Regime Formation Documentation](docs/Market_Regime_Formation.md).

1. **Directional Component**:
   * Calculated by combining the directional biases from Trending OI with PA, EMA Indicators, VWAP Indicators, and Greek Sentiment Analysis.
   * Represents the overall bullish or bearish sentiment in the market.
   * **Implementation**: Weighted sum of indicator scores in `src/market_regime_classifier.py`.

2. **Volatility Component**:
   * Calculated by combining the volatility assessments from IV Skew Analysis and ATR Indicators.
   * Represents the overall level of market volatility.
   * **Implementation**: Threshold-based classification in `src/market_regime_classifier.py`.

3. **Regime Assignment**:
   * The directional and volatility components are used to classify the market into one of 18 predefined regimes.
   * The classification is based on predefined thresholds and rules.
   * **Implementation**: Rule-based classification in `src/market_regime_classifier.py`.

   **The 18 Market Regime Types, as defined in `market_regime_18_types_config.json`, are:**

   * **Bullish Regimes:**
       * Strong_Bullish
       * Bullish
       * Moderately_Bullish
       * Weakly_Bullish
       * Bullish_Consolidation
       * Bullish_Volatile

   * **Neutral/Sideways Regimes:**
       * Neutral_Bullish_Bias
       * Neutral
       * Neutral_Bearish_Bias

   * **Bearish Regimes:**
       * Bearish_Volatile
       * Bearish_Consolidation
       * Weakly_Bearish
       * Moderately_Bearish
       * Bearish
       * Strong_Bearish

   * **Input**: `data/market_data`
   * **Output**: `data/output/market_data`

#### Dynamic Adjustments

1. **Time-Based Adjustments**:
   * Indicator weights are adjusted based on the time of day (opening, mid-day, closing).
   * This accounts for the changing dynamics of the market throughout the day.
   * **Implementation**: Time-of-day weights in `src/market_regime_classifier.py`.

2. **Performance-Based Adjustments**:
   * Indicator weights and classification thresholds are adjusted based on the historical performance of each indicator.
   * This allows the system to adapt to changing market conditions and improve accuracy.
   * **Implementation**: Weight optimization in `src/market_regime_classifier.py`.

3. **Instrument-Based Dynamic Weighting**:
   * More weightage is given to ATM straddle, ATM CE, and ATM PE signals than to underlying price signals.
   * The weightage between different instruments is dynamically adjusted based on market volatility, time of day, and recent performance.
   * **Implementation**: Dynamic weight adjustment in `utils/dynamic_weight_adjustment/dynamic_weight_adjustment.py`.
   * This ensures that the most relevant market signals are prioritized in different market conditions.

#### Output Format

The market regime data is saved to CSV files in the `data/output/market_data` directory, following the naming convention `<symbol>.market_regime_output_DDMMYYHHMM.CSV`, where `<symbol>` is the market symbol (e.g., NIFTY) and `DDMMYYHHMM` is the date and time of the file creation.

**Columns:**

1. **Date**: Trading date in YYYY-MM-DD format
2. **Zone**: Trading zone identifier
3. **Day**: Day of the week
4. **Time**: Trading time in HH:MM:SS format
5. **DTE**: Days to expiration
6. **Indicator Columns**:
   * Greek_Sentiment: Greek sentiment classification
   * Greek_Sentiment_Score: Numerical score (-1 to +1)
   * Greek_Confidence: Confidence in Greek sentiment (0 to 1)
   * Trending_OI_PA: OI/PA trend classification
   * Trending_OI_PA_Score: Numerical score (-1 to +1)
   * Trending_OI_PA_Confidence: Confidence in OI/PA trend (0 to 1)
   * IV_Skew_Classification: IV skew classification
   * IV_Skew_Score: Numerical score (-1 to +1)
   * IV_Skew_Confidence: Confidence in IV skew (0 to 1)
   * EMA_Signal: EMA signal classification
   * EMA_Score: Numerical score (-1 to +1)
   * EMA_Confidence: Confidence in EMA signal (0 to 1)
   * VWAP_Signal: VWAP signal classification
   * VWAP_Score: Numerical score (-1 to +1)
   * VWAP_Confidence: Confidence in VWAP signal (0 to 1)
   * ATR_Volatility: ATR volatility classification
   * ATR_Score: Numerical score (0 to +1)
7. **Composite Columns**:
   * Directional_Component: Combined directional score (-1 to +1)
   * Directional_Confidence: Confidence in directional component (0 to 1)
   * Volatility_Component: Combined volatility score (0 to +1)
   * Volatility_Confidence: Confidence in volatility component (0 to 1)
8. **Transition Columns**:
   * Transition_Type: Type of transition if detected
   * Transition_Probability: Probability of regime transition (0 to 1)
   * Early_Transition_Indication: Early signs of market regime transition (e.g., "Divergence in OI and Price", "Acceleration in EMA Crossover")
9. **Final Regime Columns**:
   * Market_Regime: Classified market regime
   * Market_Regime_Confidence: Confidence in regime classification (0 to 1)

### 2. Consolidation

The consolidation process, implemented in `core/consolidation.py`, combines strategy data processing and market regime assignment. This step takes the market regime classifications from the previous step and combines them with strategy performance data.

For detailed information on the consolidation process, see [Consolidation Documentation](docs/Consolidation.md).

**Input:**

* `data/input/TV_Zone_Files`: TradingView zone files
* `data/input/Python_Multi_Zone_Files`: Python multi-zone files
* `data/output/market_data`: Market regime output file

**Output Format:**

| Column                  | Description                                  |
| ----------------------- | -------------------------------------------- |
| Date                    | Trading date in YYYY-MM-DD format             |
| Zone                    | Trading zone identifier                      |
| Day                     | Day of the week                               |
| Time                    | Trading time in HH:MM:SS format             |
| Market regime confidence | Confidence score                             |
| Market regime           | Market regime classification                 |
| Market transition       | Market regime early transition indication    |
| DTE                     | Days to expiration                           |
| strategy1               | Performance of strategy 1                    |
| strategy2               | Performance of strategy 2                    |
| strategy3               | Performance of strategy 3                    |

### 3. Dimension Selection

This step, implemented in `core/dimension_selection.py`, selects the most relevant dimensions (DTE, Zone, market regime, day) for optimization. It also considers how sets of strategies or individual strategies fit into these dimensions.

For detailed information on the dimension selection process, see [Dimension Selection Documentation](docs/Dimension_Selection.md).

The process involves:

* Identifying available dimensions in the consolidated data.
* Calculating the correlation between each dimension and the strategy PnL.
* Selecting the dimensions with the highest correlation.
* Creating combined selections of dimensions for optimization.

### 4. Optimization

This step, implemented in `core/optimization.py`, runs optimization algorithms to find the optimal parameters for each strategy based on the selected dimensions.

For detailed information on the optimization process, see [Optimization Documentation](docs/Optimization.md).

The process involves:

* Running multiple optimization algorithms to find the best combinations of parameters:
  * Differential Evolution
  * Hill Climbing
  * Genetic Algorithm
  * Particle Swarm
  * Simulated Annealing
  * Ant Colony
  * Bayesian Optimization
  * Custom Differential Evolution
* Defining an optimization target (e.g., PnL) and direction (maximize or minimize).
* Finding the best results across all optimizations.

### 5. Results Visualization

This step, implemented in `core/results_visualization.py`, generates visualizations of the optimization results, including equity curves and algorithm comparisons.

For detailed information on the results visualization process, see [Results Visualization Documentation](docs/Results_Visualization.md).

### 6. PostgreSQL Integration

The final output, including the optimized parameters and performance metrics, is saved to a PostgreSQL database using `core/integration/postgresql_integration.py`.

For detailed information on the PostgreSQL integration, see [PostgreSQL Integration Documentation](docs/PostgreSQL_Integration.md).

### 7. GDFL Live Data Feed Integration

The system integrates with a GDFL data feed to obtain real-time market data and identify the current market regime. This allows for dynamic adjustment of strategies based on the prevailing market conditions.

For detailed information on the GDFL live data feed integration, see [GDFL Live Data Feed Documentation](docs/GDFL_Live_Data_Feed.md).

## Running the Pipeline

The main entry point for running the pipeline is `unified_market_regime_pipeline.py`. This script orchestrates the entire process from data input to results visualization.

For detailed information on how to run the pipeline, see [Unified Market Regime Pipeline Documentation](docs/Unified_Market_Regime_Pipeline.md).

## Directory Structure

```
enhanced-market-regime-optimizer/
├── unified_market_regime_pipeline.py  # Main entry point for the pipeline
├── batch_files/           # Batch files for running the pipeline
│   ├── run_prod_market_regime.bat
│   └── run_test_market_regime.bat
├── config/                # Configuration files
│   ├── config.py
│   ├── local_config.ini
│   ├── pipeline_config.ini
│   └── test_config.ini
├── core/                  # Core pipeline components
│   ├── consolidation.py   # Combines strategy data and market regime
│   ├── dimension_selection.py # Selects dimensions for optimization
│   ├── optimization.py    # Runs optimization algorithms
│   ├── results_visualization.py # Generates visualizations of results
│   ├── algorithms/        # Optimization algorithms
│   │   ├── differential_evolution.py
│   │   ├── hill_climbing.py
│   │   ├── genetic_algorithm.py
│   │   ├── particle_swarm.py
│   │   ├── simulated_annealing.py
│   │   ├── ant_colony.py
│   │   ├── bayesian.py
│   │   └── custom_differential_evolution.py
│   └── integration/
│       └── postgresql_integration.py # Saves output to PostgreSQL
├── data/                  # Data directory
│   ├── input/             # Input data
│   │   ├── TV_Zone_Files/       # TradingView zone files
│   │   └── Python_Multi_Zone_Files/ # Python multi-zone files
│   └── market_data/       # Market data
├── output/                # Output directory
├── src/                   # Source code
│   ├── config_manager.py
│   ├── indicator_factory.py
│   ├── market_regime_classifier.py
│   └── minute_regime_classifier.py
├── utils/                 # Utility functions
│   ├── helpers.py         # Helper functions
│   ├── market_indicators.py # Market indicator calculations
│   ├── market_regime_naming.py # Market regime naming conventions
│   ├── dynamic_weight_adjustment/ # Dynamic weight adjustment for indicators
│   │   ├── __init__.py
│   │   └── dynamic_weight_adjustment.py # Adjusts weights based on market conditions
│   └── feature_engineering/
│       ├── __init__.py
│       ├── base.py
│       ├── dte_config.py
│       ├── atr_indicators/
│       │   └── __init__.py
│       ├── config/
│       │   └── day_trader_config.ini
│       ├── ema_indicators/
│       │   ├── __init__.py
│       │   └── ema_indicators.py
│       ├── greek_sentiment/
│       │   ├── __init__.py
│       │   ├── data_processor.py
│       │   ├── dynamic_weight_adjuster.py
│       │   ├── greek_sentiment_analysis.py
│       │   └── greek_regime_classifier.py
│       ├── iv_skew/
│       │   └── iv_skew_analysis.py
│       ├── trending_oi_pa/
│       │   ├── __init__.py
│       │   └── trending_oi_pa_analysis.py
│       └── vwap_indicators/
│           ├── __init__.py
│           └── vwap_indicators.py
├── docs/                  # Documentation
│   ├── Unified_Market_Regime_Pipeline.md
│   ├── Market_Regime_Formation.md
│   ├── Consolidation.md
│   ├── Dimension_Selection.md
│   ├── Results_Visualization.md
│   ├── PostgreSQL_Integration.md
│   ├── GDFL_Live_Data_Feed.md
│   ├── Trending_OI_PA.md
│   ├── EMA_VWAP_Indicators.md
│   ├── Greek_sentiment.md
│   └── IV_Skew_Percentile.md
└── tests/                 # Test files
```

## Source Code Repository

The source code is available on GitHub: https://github.com/maruthakumar/enhanced-market-regime-optimizer.git
