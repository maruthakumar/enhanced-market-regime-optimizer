#!/bin/bash

# PostgreSQL Database Initialization Script
# This script creates the necessary database and tables for the market regime system

# Configuration
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="market_regime"
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_SCHEMA="public"

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database already exists"

# Connect to the database and create tables
echo "Creating tables..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF

-- Create market data table
CREATE TABLE IF NOT EXISTS ${DB_SCHEMA}.mr_market_data (
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
);

-- Create options data table
CREATE TABLE IF NOT EXISTS ${DB_SCHEMA}.mr_options_data (
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
);

-- Create market regimes table
CREATE TABLE IF NOT EXISTS ${DB_SCHEMA}.mr_market_regimes (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    regime VARCHAR(20) NOT NULL,
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
);

-- Create optimization results table
CREATE TABLE IF NOT EXISTS ${DB_SCHEMA}.mr_optimization_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    fitness_value DECIMAL(10,4) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON ${DB_SCHEMA}.mr_market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_options_data_symbol_timestamp ON ${DB_SCHEMA}.mr_options_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_regimes_symbol_timestamp ON ${DB_SCHEMA}.mr_market_regimes(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_optimization_results_symbol ON ${DB_SCHEMA}.mr_optimization_results(symbol);

EOF

echo "Database initialization completed successfully!"
