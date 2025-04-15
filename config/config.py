"""
Configuration file for the Enhanced Market Regime Optimizer.
"""

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    'use_dynamic_weights': True,
    'window_size': 30,
    
    # Trending OI with PA settings
    'strikes_above_atm': 7,
    'strikes_below_atm': 7,
    'oi_increase_threshold': 0.05,
    'oi_decrease_threshold': -0.05,
    'price_increase_threshold': 0.01,
    'price_decrease_threshold': -0.01,
    'short_window': 5,
    'medium_window': 15,
    'long_window': 30,
    'strong_trend_threshold': 0.10,
    'weak_trend_threshold': 0.05,
    'high_velocity_threshold': 0.03,
    'high_acceleration_threshold': 0.01,
    'institutional_lot_size': 100,
    'pattern_lookback': 20,
    'pattern_similarity_threshold': 0.80,
    'learning_rate': 0.1,
    'history_window': 60,
    'pattern_performance_lookback': 5,
    'pattern_history_file': 'pattern_history.pkl',
    'min_pattern_occurrences': 10,
    'divergence_threshold': 0.3,
    'divergence_window': 10,
    
    # Greek sentiment settings
    'delta_threshold': 0.5,
    'gamma_threshold': 0.05,
    'theta_threshold': -0.01,
    'vega_threshold': 0.1,
    'delta_weight': 0.4,
    'gamma_weight': 0.2,
    'theta_weight': 0.2,
    'vega_weight': 0.2,
    
    # IV skew settings
    'iv_percentile_lookback': 30,
    'iv_skew_threshold': 0.1,
    'iv_term_structure_lookback': 5,
    
    # EMA indicator settings
    'ema_short': 20,
    'ema_medium': 100,
    'ema_long': 200,
    
    # VWAP indicator settings
    'vwap_lookback': 1,
    
    # Multi-timeframe settings
    'timeframes': [15, 10, 5, 3],  # minutes
    'timeframe_weights': {
        15: 0.4,
        10: 0.3,
        5: 0.2,
        3: 0.1
    },
    
    # Time-of-day settings
    'pre_market_start': '08:45:00',
    'pre_market_end': '09:15:00',
    'opening_range_start': '09:15:00',
    'opening_range_end': '09:45:00',
    'active_trading_start': '09:45:00',
    'active_trading_end': '14:30:00',
    'closing_phase_start': '14:30:00',
    'closing_phase_end': '15:30:00',
    'time_of_day_weights': {
        'pre_market': 0.1,
        'opening_range': 0.3,
        'active_trading': 0.4,
        'closing_phase': 0.2
    },
    
    # Performance metrics settings
    'performance_lookback': 30,
    'min_trades_for_metrics': 10,
    'risk_free_rate': 0.05,
    
    # Visualization settings
    'plot_width': 12,
    'plot_height': 8,
    'dpi': 100,
    'color_scheme': 'viridis',
    
    # Logging settings
    'log_level': 'INFO',
    'log_file': 'market_regime_optimizer.log',
    
    # File paths
    'data_dir': './data',
    'output_dir': './output',
    'model_dir': './models',
    'visualization_dir': './visualizations'
}

# Market regime definitions
MARKET_REGIMES = [
    'Strong_Bullish',
    'Bullish',
    'Mild_Bullish',
    'Sideways_To_Bullish',
    'Neutral',
    'Sideways_To_Bearish',
    'Mild_Bearish',
    'Bearish',
    'Strong_Bearish',
    'Reversal_Imminent_Bullish',
    'Reversal_Imminent_Bearish',
    'Exhaustion_Bullish',
    'Exhaustion_Bearish',
    'Failed_Breakout_Bullish',
    'Failed_Breakout_Bearish',
    'Institutional_Accumulation',
    'Institutional_Distribution',
    'Choppy'
]

# Indicator weights for market regime identification
INDICATOR_WEIGHTS = {
    'trending_oi_pa': 0.30,
    'greek_sentiment': 0.20,
    'iv_skew': 0.15,
    'ema_indicators': 0.15,
    'vwap_indicators': 0.10,
    'ce_pe_percentile': 0.10
}

# OI pattern definitions
OI_PATTERNS = {
    'Long_Build_Up': 'OI increases + Price increases (calls) or OI increases + Price decreases (puts)',
    'Short_Build_Up': 'OI increases + Price decreases (calls) or OI increases + Price increases (puts)',
    'Long_Unwinding': 'OI decreases + Price decreases (calls) or OI decreases + Price increases (puts)',
    'Short_Covering': 'OI decreases + Price increases (calls) or OI decreases + Price decreases (puts)'
}

# Combined pattern definitions
COMBINED_PATTERNS = {
    'Strong_Bullish': 'Call Long Build-Up + Put Short Build-Up/Unwinding or Call Short Covering + Put Short Build-Up/Unwinding',
    'Mild_Bullish': 'Call Long Build-Up or Put Long Unwinding or Put Short Covering',
    'Strong_Bearish': 'Put Long Build-Up + Call Short Build-Up/Unwinding or Put Short Covering + Call Short Build-Up/Unwinding',
    'Mild_Bearish': 'Put Long Build-Up or Call Long Unwinding or Call Short Build-Up',
    'Neutral': 'Balanced or conflicting patterns'
}

# Required data columns
REQUIRED_COLUMNS = [
    'datetime',
    'strike',
    'option_type',
    'open_interest',
    'price',
    'underlying_price',
    'volume',
    'iv',
    'delta',
    'gamma',
    'theta',
    'vega'
]

# Alternative column names mapping
COLUMN_MAPPING = {
    'datetime': ['timestamp', 'date_time', 'time'],
    'strike': ['Strike', 'strike_price', 'STRIKE'],
    'option_type': ['type', 'call_put', 'cp', 'option_type'],
    'open_interest': ['OI', 'OPEN_INTEREST', 'oi'],
    'price': ['close', 'Close', 'CLOSE', 'last_price'],
    'underlying_price': ['underlying', 'Underlying', 'spot_price', 'index_price'],
    'volume': ['Volume', 'VOLUME', 'vol'],
    'iv': ['IV', 'implied_volatility', 'impliedVolatility'],
    'delta': ['Delta', 'DELTA'],
    'gamma': ['Gamma', 'GAMMA'],
    'theta': ['Theta', 'THETA'],
    'vega': ['Vega', 'VEGA']
}
