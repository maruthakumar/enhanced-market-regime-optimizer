"""
Feature Engineering Package

This package contains modules for feature engineering.

Modules:
- ema_indicators: EMA 20, 100, 200 on ATM straddle & ATM CE/PE
- atr_indicators: ATR percentile, ATR/EMA ratio
- iv_indicators: IV percentile, IV skew
- premium_indicators: ATM straddle premium percentile, ATM CE/PE premium percentile
- trending_oi_pa: Trending Open Interest with Price Action
- greek_sentiment: Greek sentiment analysis
- vwap_indicators: VWAP and previous day's VWAP
- volume_indicators: Volume analysis

Each module can be used independently or combined for comprehensive feature engineering.
"""

from .base import FeatureBase, register_feature, cache_result, time_execution
