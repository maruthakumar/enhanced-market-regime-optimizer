# Enhanced Market Regime Optimizer

A sophisticated trading analysis system that processes market data to identify different market regimes using Greek sentiment analysis, technical indicators, and machine learning techniques. This system helps traders optimize their strategies based on current market conditions.

## Complete Optimizer Pipeline

The Enhanced Market Regime Optimizer operates through a comprehensive pipeline with several interconnected components:

### 1. Data Ingestion
- **Market Data**: Price, volume, and technical indicators from PostgreSQL database
- **Options Data**: Strike prices, Greeks (Delta, Vega, Theta), open interest, and volume
- **GDFL Data Feed Integration**: Real-time data processing with dedicated adapter

### 2. Feature Engineering
- **Greek Sentiment Analysis**:
  - Analyzes option Greeks across different strike prices (ATM, near OTM, mid OTM, far OTM)
  - Delta range from 0.5 to 0.01 for comprehensive coverage
  - Expiry weightage: 70% near expiry, 30% next expiry
  - Component weighting: Vega (40%), Delta (40%), Theta (20%)
  - DTE-specific calculations with bucketing for time sensitivity

- **Trending OI with PA**:
  - Analyzes ATM plus 7 strikes above and below (15 total strikes)
  - Correlates open interest trends with price action
  - Detects accumulation/distribution patterns
  - Identifies institutional positioning
  - Rolling calculation for trending OI of calls and puts

- **Technical Indicators**:
  - EMA (Exponential Moving Average) calculations
  - VWAP (Volume Weighted Average Price) analysis
  - Volume profile and analysis
  - Volatility measurements (IV percentile, ATR, historical volatility)

### 3. Market Regime Classification
- **18 Distinct Market Regimes**:
  - Direction categories: Strong Bullish, Mild Bullish, Neutral, Mild Bearish, Strong Bearish
  - Volatility categories: High Volatility, Medium Volatility, Low Volatility
  - Combined to form regimes like "STRONG_BULLISH_HIGH_VOLATILITY"

- **Dynamic Indicator Weighting**:
  - Adaptive weights based on market conditions
  - Confidence scoring for each regime classification
  - Transition probability modeling between regimes

### 4. Database Integration
- **PostgreSQL Integration**:
  - 1-minute rolling market regime storage
  - Time-series optimized tables and partitioning
  - Efficient batch processing with connection pooling
  - Data retention policies for managing storage growth

### 5. Strategy Optimization
- **Regime-Specific Strategies**:
  - Customized trading approaches for each market regime
  - Risk management parameters adjusted by regime
  - Position sizing recommendations
  - Entry/exit timing optimization

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │     │Feature Engineering│    │ Market Regime   │
│  ┌───────────┐  │     │  ┌───────────┐   │    │ Classification  │
│  │Market Data│──┼────►│  │   Greek   │   │    │  ┌───────────┐  │
│  └───────────┘  │     │  │ Sentiment │   │    │  │  Regime   │  │
│  ┌───────────┐  │     │  │ Analysis  │   │    │  │Classifier │  │
│  │Options Data│──┼────►│  └───────────┘   │───►│  └───────────┘  │
│  └───────────┘  │     │  ┌───────────┐   │    │  ┌───────────┐  │
│  ┌───────────┐  │     │  │ Trending  │   │    │  │  Regime   │  │
│  │ GDFL Feed │──┼────►│  │  OI with  │   │    │  │  Naming   │  │
│  └───────────┘  │     │  │    PA     │   │    │  └───────────┘  │
└─────────────────┘     │  └───────────┘   │    └─────────────────┘
                        │  ┌───────────┐   │             │
                        │  │ Technical │   │             │
                        │  │Indicators │   │             │
                        │  └───────────┘   │             ▼
                        └─────────────────┘    ┌─────────────────┐
                                               │   PostgreSQL    │
                                               │   Integration   │
                                               │  ┌───────────┐  │
                                               │  │1-min Rolling│ │
                                               │  │  Storage   │ │
                                               │  └───────────┘  │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │    Strategy     │
                                               │   Optimization  │
                                               │  ┌───────────┐  │
                                               │  │ Regime-   │  │
                                               │  │ Specific  │  │
                                               │  │Strategies │  │
                                               │  └───────────┘  │
                                               └─────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/maruthakumar/enhanced-market-regime-optimizer.git
cd enhanced-market-regime-optimizer

# Install dependencies
pip install -r requirements.txt

# Set up database (if using PostgreSQL)
psql -U postgres -f config/database/initialize_database.sql
```

## Usage Examples

### 1. Greek Sentiment Analysis

```python
from utils.feature_engineering.greek_sentiment.greek_sentiment_analysis import GreekSentimentAnalyzer

# Initialize the analyzer
greek_analyzer = GreekSentimentAnalyzer()

# Load options data (example format)
options_data = {
    'near_expiry': {
        'calls': [
            {'strike': 95, 'delta': 0.45, 'vega': 0.15, 'theta': -0.05},
            {'strike': 100, 'delta': 0.35, 'vega': 0.18, 'theta': -0.06},
            # More strikes...
        ],
        'puts': [
            {'strike': 95, 'delta': -0.55, 'vega': 0.15, 'theta': -0.05},
            {'strike': 100, 'delta': -0.65, 'vega': 0.18, 'theta': -0.06},
            # More strikes...
        ]
    },
    'next_expiry': {
        # Similar structure as near_expiry
    }
}

# Calculate Greek sentiment
current_price = 100.0
sentiment_score, sentiment_category, details = greek_analyzer.calculate_greek_sentiment(
    options_data, 
    current_price=current_price
)

print(f"Greek Sentiment Score: {sentiment_score}")
print(f"Sentiment Category: {sentiment_category}")
print(f"Details: {details}")
```

### 2. Trending OI with PA Analysis

```python
from utils.feature_engineering.trending_oi_pa.trending_oi_pa_analysis import TrendingOIPAAnalyzer

# Initialize the analyzer
oi_analyzer = TrendingOIPAAnalyzer(strike_range=7)

# Load options data with OI information
options_data = {
    'calls': [
        {'strike': 95, 'oi': 1500, 'oi_change': 200},
        {'strike': 100, 'oi': 2500, 'oi_change': 350},
        # More strikes...
    ],
    'puts': [
        {'strike': 95, 'oi': 1800, 'oi_change': 150},
        {'strike': 100, 'oi': 2200, 'oi_change': 300},
        # More strikes...
    ]
}

# Price action data
price_data = {
    'current': 100.0,
    'previous': 99.5,
    'change_percent': 0.5,
    'volume': 1500000,
    'avg_volume': 1200000
}

# Calculate trending OI with PA
trending_oi_score, trending_oi_category, metrics = oi_analyzer.calculate_trending_oi(
    options_data, 
    price_data
)

print(f"Trending OI Score: {trending_oi_score}")
print(f"Trending OI Category: {trending_oi_category}")
print(f"Metrics: {metrics}")
```

### 3. Market Regime Classification

```python
from src.market_regime_classifier import MarketRegimeClassifier
from utils.market_regime_naming import MarketRegimeNames

# Initialize the classifier and naming utility
classifier = MarketRegimeClassifier()
regime_names = MarketRegimeNames()

# Prepare indicators
indicators = {
    'greek_sentiment': 0.75,  # Strong bullish
    'trending_oi_pa': 0.65,   # Bullish
    'technical_indicators': {
        'ema': 0.8,           # Above EMA
        'vwap': 0.7,          # Above VWAP
        'volume': 0.6         # Above average volume
    },
    'volatility_indicators': {
        'iv_percentile': 0.8, # High IV
        'atr': 0.7,           # High ATR
        'historical_vol': 0.6 # Above average volatility
    }
}

# Classify market regime
regime, confidence_score, details = classifier.classify_market_regime(indicators)

# Get regime description
description = regime_names.get_regime_description(regime)

print(f"Market Regime: {regime}")
print(f"Confidence Score: {confidence_score}")
print(f"Description: {description}")
print(f"Details: {details}")
```

### 4. PostgreSQL Integration

```python
from core.integration.postgresql_integration import PostgreSQLIntegration

# Configure database connection
config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'market_regime',
    'user': 'postgres',
    'password': 'your_password',
    'min_connections': 1,
    'max_connections': 5,
    'retention_days': 30
}

# Initialize database connection
db = PostgreSQLIntegration(config)

# Store market regime data (1-minute interval)
regime_data = {
    'timestamp': '2023-04-13 14:30:00',
    'symbol': 'SPY',
    'regime': 'STRONG_BULLISH_HIGH_VOLATILITY',
    'confidence': 0.85,
    'greek_sentiment': 0.75,
    'trending_oi_pa': 0.65,
    'technical_score': 0.70,
    'volatility_score': 0.80
}

db.store_market_regime(regime_data)

# Query recent market regimes
recent_regimes = db.get_recent_market_regimes(
    symbol='SPY',
    minutes=60,  # Last hour
    interval='1min'
)

print(f"Recent regimes: {recent_regimes}")
```

## Project Structure

```
enhanced-market-regime-optimizer/
├── config/                        # Configuration files
│   ├── database/                  # Database configuration
│   │   ├── initialize_database.sql # SQL schema initialization
├── core/                          # Core functionality
│   ├── integration/               # External integrations
│   │   ├── postgresql_integration.py  # Database integration
│   │   ├── gdfl_integration.py    # GDFL data feed integration
├── src/                           # Source code
│   ├── market_regime_classifier.py # Market regime classification
├── utils/                         # Utility modules
│   ├── feature_engineering/       # Feature engineering components
│   │   ├── greek_sentiment/       # Greek sentiment analysis
│   │   │   ├── greek_sentiment_analysis.py
│   │   ├── trending_oi_pa/        # Trending OI with PA
│   │   │   ├── trending_oi_pa_analysis.py
│   │   ├── dte_config.py          # DTE configuration
│   ├── market_regime_naming.py    # Standardized naming
├── tests/                         # Test suite
│   ├── test_greek_sentiment.py
│   ├── test_trending_oi_pa.py
│   ├── test_market_regime.py
├── docs/                          # Documentation
│   ├── CONTRIBUTING.md
│   ├── CODE_OF_CONDUCT.md
├── .github/                       # GitHub configuration
│   ├── workflows/                 # GitHub Actions workflows
│   │   ├── test.yml
│   │   ├── lint.yml
│   │   ├── release.yml
│   ├── ISSUE_TEMPLATE/           # Issue templates
│   │   ├── bug_report.md
│   │   ├── feature_request.md
├── .gitignore                     # Git ignore file
├── CHANGELOG.md                   # Version changelog
├── LICENSE                        # License file
├── README.md                      # This file
├── requirements.txt               # Python dependencies
```

## Contributing

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/maruthakumar/enhanced-market-regime-optimizer/tags).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
