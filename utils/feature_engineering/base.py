"""
Base Feature Engineering Module

This module provides the base classes and decorators for all feature engineering components.
It includes:
- FeatureBase: Base class for all feature components
- register_feature: Decorator to register feature components
- cache_result: Decorator to cache function results
- time_execution: Decorator to time function execution
- FeatureCache: Cache for feature calculations
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import configparser
from functools import wraps
import time
import concurrent.futures
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature registry
_FEATURE_REGISTRY = {}

# Cache for feature calculations
class FeatureCache:
    """Cache for feature calculations."""
    
    def __init__(self, max_size=100):
        """
        Initialize the feature cache.
        
        Args:
            max_size (int): Maximum cache size
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Value from cache or None if not found
        """
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key, value):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove first item (oldest)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get_hit_rate(self):
        """
        Get cache hit rate.
        
        Returns:
            float: Cache hit rate
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


# Decorators
def cache_result(func):
    """
    Decorator to cache function results.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip caching if cache is not enabled
        if not hasattr(self, 'cache') or not self.use_cache:
            return func(self, *args, **kwargs)
        
        # Create cache key
        key = (func.__name__, str(args), str(kwargs))
        
        # Check cache
        result = self.cache.get(key)
        if result is not None:
            return result
        
        # Calculate result
        result = func(self, *args, **kwargs)
        
        # Cache result
        self.cache.set(key, result)
        
        return result
    
    return wrapper


def time_execution(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.6f} seconds")
        
        return result
    
    return wrapper


# Base class for feature components
class FeatureBase:
    """Base class for feature components."""
    
    def __init__(self, config=None):
        """
        Initialize the feature component.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        
        # Set default configuration values
        self.use_cache = self.config.get('use_cache', True)
        self.cache_size = int(self.config.get('cache_size', 100))
        
        # Initialize cache
        if self.use_cache:
            self.cache = FeatureCache(max_size=self.cache_size)
    
    @time_execution
    def calculate_features(self, data_frame, **kwargs):
        """
        Calculate features.
        
        Args:
            data_frame (pd.DataFrame): Input data
            **kwargs: Additional arguments
            
        Returns:
            pd.DataFrame: Data with calculated features
        """
        raise NotImplementedError("Subclasses must implement calculate_features")
    
    def clear_cache(self):
        """Clear cache."""
        if hasattr(self, 'cache'):
            self.cache.clear()
            logger.info(f"Cleared cache for {self.__class__.__name__}")
    
    def get_cache_hit_rate(self):
        """
        Get cache hit rate.
        
        Returns:
            float: Cache hit rate
        """
        if hasattr(self, 'cache'):
            return self.cache.get_hit_rate()
        return 0.0


# Feature registry functions
def register_feature(name=None, category=None):
    """
    Decorator to register a feature class.
    
    Args:
        name (str, optional): Feature name
        category (str, optional): Feature category
    
    Returns:
        callable: Decorator function
    """
    def decorator(cls):
        cls._feature_name = name or cls.__name__
        cls._feature_category = category or 'general'
        return cls
    return decorator


def get_feature(name, category=None):
    """
    Get a feature component by name.
    
    Args:
        name (str): Feature name
        category (str, optional): Feature category
        
    Returns:
        Feature component class
    """
    # If category is provided, look only in that category
    if category is not None:
        if category not in _FEATURE_REGISTRY:
            logger.error(f"Feature category {category} not found")
            return None
        
        if name not in _FEATURE_REGISTRY[category]:
            logger.error(f"Feature {name} not found in category {category}")
            return None
        
        return _FEATURE_REGISTRY[category][name]
    
    # If category is not provided, search all categories
    for category, features in _FEATURE_REGISTRY.items():
        if name in features:
            return features[name]
    
    logger.error(f"Feature {name} not found")
    return None


def get_all_features():
    """
    Get all registered features.
    
    Returns:
        dict: Dictionary of all registered features
    """
    return _FEATURE_REGISTRY
