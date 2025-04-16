# PostgreSQL Integration

## Overview

The PostgreSQL Integration module is a critical component of the Enhanced Market Regime Optimizer pipeline that handles the storage and retrieval of market data, options data, market regimes, and optimization results in a PostgreSQL database. This module provides efficient and reliable data persistence, enabling historical analysis, backtesting, and real-time data processing.

The PostgreSQL integration is implemented in `core/integration/postgresql_integration.py` and serves as the data storage layer for the entire pipeline. It ensures that all generated data is properly stored and can be easily retrieved for future analysis.

## Purpose and Importance

PostgreSQL integration is crucial for several reasons:

1. **Data Persistence**: Ensures that all generated data is stored persistently and can be retrieved even after the application is restarted.

2. **Historical Analysis**: Enables historical analysis of market regimes, strategy performance, and optimization results.

3. **Efficient Queries**: Provides efficient query capabilities for retrieving specific data based on various criteria.

4. **Time-Series Optimization**: Includes specific optimizations for time-series data, which is critical for market data and market regimes.

5. **Scalability**: Allows the system to handle large volumes of data efficiently.

## Database Schema

The PostgreSQL integration creates and manages the following tables:

### 1. Market Data Table

Stores market data for various symbols and timeframes.

```sql
CREATE TABLE IF NOT EXISTS {schema}.{market_data_table} (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
)
```

### 2. Options Data Table

Stores options data for various symbols, expiry dates, strike prices, and option types.

```sql
CREATE TABLE IF NOT EXISTS {schema}.{options_data_table} (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    expiry_date DATE NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    option_type VARCHAR(4) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    open_interest BIGINT NOT NULL,
    implied_volatility DECIMAL(10,4),
    delta DECIMAL(10,4),
    gamma DECIMAL(10,4),
    theta DECIMAL(10,4),
    vega DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, expiry_date, strike_price, option_type, timestamp)
)
```

### 3. Market Regimes Table

Stores market regime classifications for various symbols and timeframes.

```sql
CREATE TABLE IF NOT EXISTS {schema}.{market_regimes_table} (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    regime VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    dte_value DECIMAL(10,4),
    atr_value DECIMAL(10,4),
    ema_value DECIMAL(10,4),
    volume_value DECIMAL(10,4),
    vwap_value DECIMAL(10,4),
    greek_sentiment DECIMAL(10,4),
    iv_value DECIMAL(10,4),
    premium_value DECIMAL(10,4),
    oi_trend_value DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
)
```

### 4. 1-Minute Market Regimes Table

Stores 1-minute rolling market regime data with time-series optimization.

```sql
CREATE TABLE IF NOT EXISTS {schema}.{market_regimes_1min_table} (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    regime VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    directional_component DECIMAL(10,4),
    volatility_component DECIMAL(10,4),
    directional_confidence DECIMAL(5,2),
    volatility_confidence DECIMAL(5,2),
    greek_sentiment VARCHAR(50),
    trending_oi_pa VARCHAR(50),
    ema_signal DECIMAL(10,4),
    vwap_signal DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
)
```

### 5. Optimization Results Table

Stores optimization results for various algorithms and parameters.

```sql
CREATE TABLE IF NOT EXISTS {schema}.{optimization_results_table} (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    fitness_value DECIMAL(10,4) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Connection Management

The PostgreSQL integration includes robust connection management to ensure efficient and reliable database access:

### Connection Pooling

The module uses connection pooling to improve performance by reusing database connections instead of creating new ones for each operation.

### Context Manager for Connections

The module provides a context manager for database connections, which ensures that connections are properly acquired and released.

## Data Storage Capabilities

The PostgreSQL integration provides several methods for storing different types of data:

### 1. Market Data Storage

Stores market data for various symbols and timeframes.

### 2. Options Data Storage

Stores options data for various symbols, expiry dates, strike prices, and option types.

### 3. Market Regime Storage

Stores market regime classifications for various symbols and timeframes.

### 4. 1-Minute Market Regime Storage

Stores 1-minute rolling market regime data with time-series optimization.

### 5. Optimization Results Storage

Stores optimization results for various algorithms and parameters.

## Data Retrieval Capabilities

The PostgreSQL integration provides several methods for retrieving different types of data:

### 1. Get 1-Minute Market Regimes

Retrieves 1-minute market regime data for a specific symbol and time range.

### 2. Get Latest 1-Minute Market Regime

Retrieves the latest 1-minute market regime for a specific symbol.

### 3. Get Market Regime Distribution

Retrieves the distribution of market regimes for a specific symbol and time range.

## Rolling Market Regime Storage

The PostgreSQL integration includes a special method for storing rolling market regimes, which is particularly useful for high-frequency data.

## Data Maintenance

The PostgreSQL integration includes a method for cleaning up old data to prevent the database from growing too large.

## Configuration Options

The PostgreSQL integration can be configured through an INI file or a configuration dictionary:

### INI File Configuration

```ini
[DATABASE]
enabled = true
host = localhost
port = 5432
database = market_regime
user = postgres
password = postgres
schema = public
table_prefix = mr_

[TABLES]
market_data = market_data
options_data = options_data
market_regimes = market_regimes
1min_market_regimes = market_regimes_1min
optimization_results = optimization_results

[CONNECTION]
max_connections = 5
connection_timeout = 30
retry_attempts = 3
retry_delay = 5
```

### Configuration Dictionary

```python
config = {
    "database": {
        "enabled": True,
        "host": "localhost",
        "port": 5432,
        "database": "market_regime",
        "user": "postgres",
        "password": "postgres",
        "schema": "public",
        "table_prefix": "mr_"
    },
    "tables": {
        "market_data": "market_data",
        "options_data": "options_data",
        "market_regimes": "market_regimes",
        "1min_market_regimes": "market_regimes_1min",
        "optimization_results": "optimization_results"
    },
    "connection": {
        "max_connections": 5,
        "connection_timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 5
    }
}
```

## Integration with the Unified Pipeline

The PostgreSQL integration is integrated with the unified pipeline through the `PostgreSQLIntegration` class, which is instantiated by the unified pipeline and used to store and retrieve data.

```python
# Initialize PostgreSQL integration
postgresql = PostgreSQLIntegration(config_file='config/database/database_config.ini')

# Initialize database
postgresql.initialize_database()

# Save market regime data
postgresql.save_market_regime(market_regime_data)

# Save 1-minute market regime data
postgresql.save_1min_market_regime(market_regime_1min_data)

# Save optimization results
postgresql.save_optimization_result(optimization_result)

# Get 1-minute market regimes
market_regimes = postgresql.get_1min_market_regimes(symbol, start_date, end_date)

# Get latest 1-minute market regime
latest_regime = postgresql.get_latest_1min_market_regime(symbol)

# Get market regime distribution
regime_distribution = postgresql.get_market_regime_distribution(symbol, start_date, end_date)

# Clean up old data
postgresql.cleanup_old_data(days_to_keep=30)

# Close connection
postgresql.close()
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Connection errors

**Symptoms**: The PostgreSQL integration fails to connect to the database.

**Solutions**:
- Check that the database server is running.
- Verify that the connection parameters (host, port, database, user, password) are correct.
- Ensure that the user has the necessary permissions to access the database.
- Check for network issues that might prevent the connection.

#### Issue: Data insertion errors

**Symptoms**: The PostgreSQL integration fails to insert data into the database.

**Solutions**:
- Check that the data being inserted matches the table schema.
- Verify that the user has the necessary permissions to insert data.
- Check for unique constraint violations.
- Ensure that the database has enough disk space.

#### Issue: Query errors

**Symptoms**: The PostgreSQL integration fails to execute queries.

**Solutions**:
- Check that the query syntax is correct.
- Verify that the user has the necessary permissions to execute the query.
- Check for missing tables or columns.
- Ensure that the query parameters are of the correct type.

### Logging and Debugging

The PostgreSQL integration includes comprehensive logging to help diagnose issues. By default, logs are written to the console and can be configured to write to a file.

To enable more detailed logging, you can adjust the logging level:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("postgresql_debug.log"),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

### Optimizing for Speed

To optimize the PostgreSQL integration for speed, consider the following:

1. **Use connection pooling**: Connection pooling reduces the overhead of creating new database connections.

2. **Use batch operations**: Batch operations reduce the number of database round-trips.

3. **Use indexes**: Indexes improve query performance, especially for time-series data.

4. **Optimize queries**: Write efficient queries that minimize the amount of data transferred.

### Optimizing for Memory Usage

To optimize the PostgreSQL integration for memory usage, consider the following:

1. **Limit result sets**: Limit the number of rows returned by queries to reduce memory usage.

2. **Stream results**: Stream query results instead of loading them all into memory.

3. **Clean up old data**: Regularly clean up old data to prevent the database from growing too large.

4. **Use appropriate data types**: Use appropriate data types to minimize storage requirements.

## Conclusion

The PostgreSQL integration is a critical component of the Enhanced Market Regime Optimizer pipeline that provides efficient and reliable data persistence. By properly configuring and using the PostgreSQL integration, you can store and retrieve market data, options data, market regimes, and optimization results, enabling historical analysis, backtesting, and real-time data processing.

For more information on other components of the pipeline, refer to the following documentation:

- [Unified Market Regime Pipeline](Unified_Market_Regime_Pipeline.md)
- [Market Regime Formation](Market_Regime_Formation.md)
- [Consolidation](Consolidation.md)
- [Dimension Selection](Dimension_Selection.md)
- [Results Visualization](Results_Visualization.md)
- [GDFL Live Data Feed](GDFL_Live_Data_Feed.md)
