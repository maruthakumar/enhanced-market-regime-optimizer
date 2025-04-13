"""
Enhanced PostgreSQL Integration Module for 1-Minute Rolling Market Regime Storage

This module implements enhanced PostgreSQL integration with specific support for
1-minute rolling market regime storage, optimized for high-frequency data.

Features:
- Efficient storage for 1-minute market regime data
- Time-series optimized tables and queries
- Connection pooling for better performance
- Batch processing for high-frequency data
"""

import psycopg2
import logging
import json
import configparser
import os
import pandas as pd
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timedelta
from psycopg2 import pool

# Setup logging
logger = logging.getLogger(__name__)

class PostgreSQLIntegration:
    """
    Enhanced PostgreSQL Integration class for the Market Regime system.
    Handles database connections, queries, and data operations with specific
    support for 1-minute rolling market regime storage.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the PostgreSQL integration with configuration.
        
        Args:
            config_file (str): Path to the database configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config', 'database', 'database_config.ini'
        )
        self.config = self._load_config()
        self.enabled = self.config.getboolean('DATABASE', 'enabled', fallback=False)
        
        if self.enabled:
            self.host = self.config.get('DATABASE', 'host')
            self.port = self.config.get('DATABASE', 'port')
            self.database = self.config.get('DATABASE', 'database')
            self.user = self.config.get('DATABASE', 'user')
            self.password = self.config.get('DATABASE', 'password')
            self.schema = self.config.get('DATABASE', 'schema')
            self.table_prefix = self.config.get('DATABASE', 'table_prefix')
            
            # Table names
            self.market_data_table = f"{self.table_prefix}{self.config.get('TABLES', 'market_data')}"
            self.options_data_table = f"{self.table_prefix}{self.config.get('TABLES', 'options_data')}"
            self.market_regimes_table = f"{self.table_prefix}{self.config.get('TABLES', 'market_regimes')}"
            self.market_regimes_1min_table = f"{self.table_prefix}{self.config.get('TABLES', '1min_market_regimes', 'market_regimes_1min')}"
            self.optimization_results_table = f"{self.table_prefix}{self.config.get('TABLES', 'optimization_results')}"
            
            # Connection settings
            self.max_connections = self.config.getint('CONNECTION', 'max_connections', fallback=5)
            self.connection_timeout = self.config.getint('CONNECTION', 'connection_timeout', fallback=30)
            self.retry_attempts = self.config.getint('CONNECTION', 'retry_attempts', fallback=3)
            self.retry_delay = self.config.getint('CONNECTION', 'retry_delay', fallback=5)
            
            # Initialize connection pool
            self._initialize_connection_pool()
            
            self.logger.info(f"PostgreSQL integration initialized with database: {self.database}")
        else:
            self.logger.info("PostgreSQL integration is disabled in configuration")
    
    def _load_config(self):
        """Load configuration from the INI file"""
        config = configparser.ConfigParser()
        try:
            config.read(self.config_file)
            return config
        except Exception as e:
            self.logger.error(f"Error loading database configuration: {str(e)}")
            # Return a default config with database disabled
            config['DATABASE'] = {'enabled': 'false'}
            return config
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for better performance"""
        if not self.enabled:
            return
            
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                1, self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=self.connection_timeout
            )
            self.logger.info(f"Initialized connection pool with {self.max_connections} max connections")
        except Exception as e:
            self.logger.error(f"Error initializing connection pool: {str(e)}")
            self.connection_pool = None
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            connection: A PostgreSQL database connection
        """
        if not self.enabled:
            self.logger.warning("Attempted to get connection while PostgreSQL integration is disabled")
            yield None
            return
            
        connection = None
        try:
            # Get connection from pool if available
            if self.connection_pool:
                connection = self.connection_pool.getconn()
            else:
                # Fall back to direct connection
                connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    connect_timeout=self.connection_timeout
                )
            yield connection
        except psycopg2.Error as e:
            self.logger.error(f"Database connection error: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                if self.connection_pool:
                    self.connection_pool.putconn(connection)
                else:
                    connection.close()
    
    def initialize_database(self):
        """
        Initialize database tables if they don't exist.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping initialize_database")
            return False
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Create market data table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.market_data_table} (
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
                """)
                
                # Create options data table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.options_data_table} (
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
                """)
                
                # Create market regimes table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.market_regimes_table} (
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
                """)
                
                # Create 1-minute market regimes table with time-series optimization
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.market_regimes_1min_table} (
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
                """)
                
                # Create optimization results table
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.optimization_results_table} (
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
                """)
                
                # Create indexes for better performance
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON {self.schema}.{self.market_data_table}(symbol, timestamp)
                """)
                
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_options_data_symbol_timestamp 
                ON {self.schema}.{self.options_data_table}(symbol, timestamp)
                """)
                
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_market_regimes_symbol_timestamp 
                ON {self.schema}.{self.market_regimes_table}(symbol, timestamp)
                """)
                
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_market_regimes_1min_symbol_timestamp 
                ON {self.schema}.{self.market_regimes_1min_table}(symbol, timestamp)
                """)
                
                # Create time-based index for 1-minute data
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_market_regimes_1min_timestamp 
                ON {self.schema}.{self.market_regimes_1min_table}(timestamp)
                """)
                
                conn.commit()
                self.logger.info("Database tables initialized successfully")
                return True
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            return False
                
    def save_market_data(self, market_data):
        """
        Save market data to the database.
        
        Args:
            market_data (dict or list): Market data to save
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping save_market_data")
            return False
            
        if isinstance(market_data, dict):
            market_data = [market_data]
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Use batch insert for better performance
                args_list = []
                for data in market_data:
                    args_list.append((
                        data['symbol'],
                        data['timestamp'],
                        data['open'],
                        data['high'],
                        data['low'],
                        data['close'],
                        data['volume'],
                        data['timeframe']
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    f"""
                    INSERT INTO {self.schema}.{self.market_data_table}
                    (symbol, timestamp, open, high, low, close, volume, timeframe)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """,
                    args_list
                )
                
                conn.commit()
                self.logger.info(f"Saved {len(market_data)} market data records to database")
                return True
        except Exception as e:
            self.logger.error(f"Error saving market data: {str(e)}")
            return False
            
    def save_options_data(self, options_data):
        """
        Save options data to the database.
        
        Args:
            options_data (dict or list): Options data to save
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping save_options_data")
            return False
            
        if isinstance(options_data, dict):
            options_data = [options_data]
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Use batch insert for better performance
                args_list = []
                for data in options_data:
                    args_list.append((
                        data['symbol'],
                        data['expiry_date'],
                        data['strike_price'],
                        data['option_type'],
                        data['timestamp'],
                        data['open'],
                        data['high'],
                        data['low'],
                        data['close'],
                        data['volume'],
                        data['open_interest'],
                        data.get('implied_volatility'),
                        data.get('delta'),
                        data.get('gamma'),
                        data.get('theta'),
                        data.get('vega')
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    f"""
                    INSERT INTO {self.schema}.{self.options_data_table}
                    (symbol, expiry_date, strike_price, option_type, timestamp, 
                     open, high, low, close, volume, open_interest, 
                     implied_volatility, delta, gamma, theta, vega)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, expiry_date, strike_price, option_type, timestamp) DO UPDATE
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        vega = EXCLUDED.vega
                    """,
                    args_list
                )
                
                conn.commit()
                self.logger.info(f"Saved {len(options_data)} options data records to database")
                return True
        except Exception as e:
            self.logger.error(f"Error saving options data: {str(e)}")
            return False
            
    def save_market_regime(self, regime_data):
        """
        Save market regime data to the database.
        
        Args:
            regime_data (dict or list): Market regime data to save
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping save_market_regime")
            return False
            
        if isinstance(regime_data, dict):
            regime_data = [regime_data]
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Use batch insert for better performance
                args_list = []
                for data in regime_data:
                    args_list.append((
                        data['symbol'],
                        data['timestamp'],
                        data['timeframe'],
                        data['regime'],
                        data['confidence'],
                        data.get('dte_value'),
                        data.get('atr_value'),
                        data.get('ema_value'),
                        data.get('volume_value'),
                        data.get('vwap_value'),
                        data.get('greek_sentiment'),
                        data.get('iv_value'),
                        data.get('premium_value'),
                        data.get('oi_trend_value')
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    f"""
                    INSERT INTO {self.schema}.{self.market_regimes_table}
                    (symbol, timestamp, timeframe, regime, confidence, 
                     dte_value, atr_value, ema_value, volume_value, vwap_value,
                     greek_sentiment, iv_value, premium_value, oi_trend_value)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE
                    SET regime = EXCLUDED.regime,
                        confidence = EXCLUDED.confidence,
                        dte_value = EXCLUDED.dte_value,
                        atr_value = EXCLUDED.atr_value,
                        ema_value = EXCLUDED.ema_value,
                        volume_value = EXCLUDED.volume_value,
                        vwap_value = EXCLUDED.vwap_value,
                        greek_sentiment = EXCLUDED.greek_sentiment,
                        iv_value = EXCLUDED.iv_value,
                        premium_value = EXCLUDED.premium_value,
                        oi_trend_value = EXCLUDED.oi_trend_value
                    """,
                    args_list
                )
                
                conn.commit()
                self.logger.info(f"Saved {len(regime_data)} market regime records to database")
                return True
        except Exception as e:
            self.logger.error(f"Error saving market regime data: {str(e)}")
            return False
    
    def save_1min_market_regime(self, regime_data):
        """
        Save 1-minute rolling market regime data to the database.
        
        Args:
            regime_data (dict or list): 1-minute market regime data to save
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping save_1min_market_regime")
            return False
            
        if isinstance(regime_data, dict):
            regime_data = [regime_data]
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Use batch insert for better performance
                args_list = []
                for data in regime_data:
                    args_list.append((
                        data['symbol'],
                        data['timestamp'],
                        data['regime'],
                        data['confidence'],
                        data.get('directional_component'),
                        data.get('volatility_component'),
                        data.get('directional_confidence'),
                        data.get('volatility_confidence'),
                        data.get('greek_sentiment'),
                        data.get('trending_oi_pa'),
                        data.get('ema_signal'),
                        data.get('vwap_signal')
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    f"""
                    INSERT INTO {self.schema}.{self.market_regimes_1min_table}
                    (symbol, timestamp, regime, confidence, 
                     directional_component, volatility_component, 
                     directional_confidence, volatility_confidence,
                     greek_sentiment, trending_oi_pa, ema_signal, vwap_signal)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO UPDATE
                    SET regime = EXCLUDED.regime,
                        confidence = EXCLUDED.confidence,
                        directional_component = EXCLUDED.directional_component,
                        volatility_component = EXCLUDED.volatility_component,
                        directional_confidence = EXCLUDED.directional_confidence,
                        volatility_confidence = EXCLUDED.volatility_confidence,
                        greek_sentiment = EXCLUDED.greek_sentiment,
                        trending_oi_pa = EXCLUDED.trending_oi_pa,
                        ema_signal = EXCLUDED.ema_signal,
                        vwap_signal = EXCLUDED.vwap_signal
                    """,
                    args_list
                )
                
                conn.commit()
                self.logger.info(f"Saved {len(regime_data)} 1-minute market regime records to database")
                return True
        except Exception as e:
            self.logger.error(f"Error saving 1-minute market regime data: {str(e)}")
            return False
    
    def save_rolling_market_regimes(self, df, symbol_column='symbol', timestamp_column='timestamp', 
                                   regime_column='Rolling_Market_Regime', confidence_column='Rolling_Market_Regime_Confidence',
                                   directional_component_column='Directional_Component', volatility_component_column='Volatility_Component',
                                   directional_confidence_column='Directional_Confidence', volatility_confidence_column='Volatility_Confidence',
                                   greek_sentiment_column='Greek_Sentiment', trending_oi_pa_column='OI_PA_Regime',
                                   ema_signal_column='EMA_Signal', vwap_signal_column='VWAP_Signal'):
        """
        Save rolling market regimes from DataFrame to database.
        
        Args:
            df (pd.DataFrame): DataFrame with market regime data
            symbol_column (str): Column name for symbol
            timestamp_column (str): Column name for timestamp
            regime_column (str): Column name for market regime
            confidence_column (str): Column name for confidence
            directional_component_column (str): Column name for directional component
            volatility_component_column (str): Column name for volatility component
            directional_confidence_column (str): Column name for directional confidence
            volatility_confidence_column (str): Column name for volatility confidence
            greek_sentiment_column (str): Column name for Greek sentiment
            trending_oi_pa_column (str): Column name for trending OI with PA
            ema_signal_column (str): Column name for EMA signal
            vwap_signal_column (str): Column name for VWAP signal
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping save_rolling_market_regimes")
            return False
            
        try:
            # Check if required columns exist
            required_columns = [symbol_column, timestamp_column, regime_column, confidence_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                return False
            
            # Prepare data for batch insert
            regime_data = []
            
            for _, row in df.iterrows():
                data = {
                    'symbol': row[symbol_column],
                    'timestamp': row[timestamp_column],
                    'regime': row[regime_column],
                    'confidence': row[confidence_column]
                }
                
                # Add optional columns if they exist
                if directional_component_column in df.columns:
                    data['directional_component'] = row[directional_component_column]
                
                if volatility_component_column in df.columns:
                    data['volatility_component'] = row[volatility_component_column]
                
                if directional_confidence_column in df.columns:
                    data['directional_confidence'] = row[directional_confidence_column]
                
                if volatility_confidence_column in df.columns:
                    data['volatility_confidence'] = row[volatility_confidence_column]
                
                if greek_sentiment_column in df.columns:
                    data['greek_sentiment'] = row[greek_sentiment_column]
                
                if trending_oi_pa_column in df.columns:
                    data['trending_oi_pa'] = row[trending_oi_pa_column]
                
                if ema_signal_column in df.columns:
                    data['ema_signal'] = row[ema_signal_column]
                
                if vwap_signal_column in df.columns:
                    data['vwap_signal'] = row[vwap_signal_column]
                
                regime_data.append(data)
            
            # Save to database
            return self.save_1min_market_regime(regime_data)
        
        except Exception as e:
            self.logger.error(f"Error saving rolling market regimes: {str(e)}")
            return False
    
    def get_1min_market_regimes(self, symbol, start_date, end_date):
        """
        Retrieve 1-minute market regime data from the database.
        
        Args:
            symbol (str): Symbol to retrieve data for
            start_date (str): Start date in ISO format
            end_date (str): End date in ISO format
            
        Returns:
            pd.DataFrame: DataFrame with 1-minute market regime data
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping get_1min_market_regimes")
            return pd.DataFrame()
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return pd.DataFrame()
                    
                query = f"""
                SELECT symbol, timestamp, regime, confidence, 
                       directional_component, volatility_component, 
                       directional_confidence, volatility_confidence,
                       greek_sentiment, trending_oi_pa, ema_signal, vwap_signal
                FROM {self.schema}.{self.market_regimes_1min_table}
                WHERE symbol = %s AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, start_date, end_date)
                )
                
                self.logger.info(f"Retrieved {len(df)} 1-minute market regime records for {symbol}")
                return df
        
        except Exception as e:
            self.logger.error(f"Error retrieving 1-minute market regime data: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_1min_market_regime(self, symbol):
        """
        Retrieve latest 1-minute market regime for a symbol.
        
        Args:
            symbol (str): Symbol to retrieve data for
            
        Returns:
            dict: Latest market regime data
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping get_latest_1min_market_regime")
            return None
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return None
                    
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT symbol, timestamp, regime, confidence, 
                           directional_component, volatility_component, 
                           directional_confidence, volatility_confidence,
                           greek_sentiment, trending_oi_pa, ema_signal, vwap_signal
                    FROM {self.schema}.{self.market_regimes_1min_table}
                    WHERE symbol = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (symbol,)
                )
                
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    result = dict(zip(columns, row))
                    self.logger.info(f"Retrieved latest 1-minute market regime for {symbol}")
                    return result
                else:
                    self.logger.info(f"No 1-minute market regime found for {symbol}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error retrieving latest 1-minute market regime: {str(e)}")
            return None
    
    def get_market_regime_distribution(self, symbol, start_date, end_date):
        """
        Get distribution of market regimes for a symbol in a date range.
        
        Args:
            symbol (str): Symbol to retrieve data for
            start_date (str): Start date in ISO format
            end_date (str): End date in ISO format
            
        Returns:
            dict: Distribution of market regimes
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping get_market_regime_distribution")
            return {}
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return {}
                    
                query = f"""
                SELECT regime, COUNT(*) as count
                FROM {self.schema}.{self.market_regimes_1min_table}
                WHERE symbol = %s AND timestamp BETWEEN %s AND %s
                GROUP BY regime
                ORDER BY count DESC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, start_date, end_date)
                )
                
                # Convert to dictionary
                distribution = {}
                for _, row in df.iterrows():
                    distribution[row['regime']] = int(row['count'])
                
                self.logger.info(f"Retrieved market regime distribution for {symbol}")
                return distribution
        
        except Exception as e:
            self.logger.error(f"Error retrieving market regime distribution: {str(e)}")
            return {}
    
    def save_optimization_result(self, optimization_data):
        """
        Save optimization result to the database.
        
        Args:
            optimization_data (dict or list): Optimization result data to save
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping save_optimization_result")
            return False
            
        if isinstance(optimization_data, dict):
            optimization_data = [optimization_data]
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Use batch insert for better performance
                args_list = []
                for data in optimization_data:
                    # Convert parameters to JSON if it's a dict
                    if isinstance(data['parameters'], dict):
                        parameters_json = json.dumps(data['parameters'])
                    else:
                        parameters_json = data['parameters']
                    
                    args_list.append((
                        data['symbol'],
                        data['algorithm'],
                        parameters_json,
                        data['fitness_value'],
                        data['start_date'],
                        data['end_date'],
                        data['timeframe']
                    ))
                
                # Execute batch insert
                cursor.executemany(
                    f"""
                    INSERT INTO {self.schema}.{self.optimization_results_table}
                    (symbol, algorithm, parameters, fitness_value, start_date, end_date, timeframe)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    args_list
                )
                
                conn.commit()
                self.logger.info(f"Saved {len(optimization_data)} optimization results to database")
                return True
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")
            return False
    
    def cleanup_old_data(self, days_to_keep=30):
        """
        Clean up old data from the database.
        
        Args:
            days_to_keep (int): Number of days of data to keep
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("PostgreSQL integration is disabled, skipping cleanup_old_data")
            return False
            
        try:
            with self.get_connection() as conn:
                if conn is None:
                    return False
                    
                cursor = conn.cursor()
                
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Delete old 1-minute market regime data
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.{self.market_regimes_1min_table}
                    WHERE timestamp < %s
                    """,
                    (cutoff_date,)
                )
                
                deleted_count = cursor.rowcount
                
                conn.commit()
                self.logger.info(f"Cleaned up {deleted_count} old 1-minute market regime records")
                return True
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            return False
    
    def close(self):
        """Close connection pool and release resources"""
        if self.enabled and self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Closed all database connections")
