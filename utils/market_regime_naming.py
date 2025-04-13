"""
Market Regime Naming Standardization Module

This module provides standardized naming for market regimes to ensure consistency
across the entire system and fix naming inconsistencies like the "voltatile" typo.

Features:
- Enumeration for market regime names to prevent typos
- Mapping functions to convert between different naming conventions
- Validation functions to ensure regime names are valid
"""

import logging
from enum import Enum, auto

# Setup logging
logger = logging.getLogger(__name__)

# Define market regime enum to prevent typos and ensure consistency
class MarketRegime(Enum):
    # Bullish regimes
    HIGH_VOLATILE_STRONG_BULLISH = auto()
    NORMAL_VOLATILE_STRONG_BULLISH = auto()
    LOW_VOLATILE_STRONG_BULLISH = auto()
    HIGH_VOLATILE_MILD_BULLISH = auto()
    NORMAL_VOLATILE_MILD_BULLISH = auto()
    LOW_VOLATILE_MILD_BULLISH = auto()
    
    # Neutral regimes
    HIGH_VOLATILE_NEUTRAL = auto()
    NORMAL_VOLATILE_NEUTRAL = auto()
    LOW_VOLATILE_NEUTRAL = auto()
    
    # Bearish regimes
    HIGH_VOLATILE_MILD_BEARISH = auto()
    NORMAL_VOLATILE_MILD_BEARISH = auto()
    LOW_VOLATILE_MILD_BEARISH = auto()
    HIGH_VOLATILE_STRONG_BEARISH = auto()
    NORMAL_VOLATILE_STRONG_BEARISH = auto()
    LOW_VOLATILE_STRONG_BEARISH = auto()
    
    # Transitional regimes
    BULLISH_TO_BEARISH_TRANSITION = auto()
    BEARISH_TO_BULLISH_TRANSITION = auto()
    VOLATILITY_EXPANSION = auto()

# Map enum to standardized string representation
REGIME_NAMES = {
    MarketRegime.HIGH_VOLATILE_STRONG_BULLISH: "High_Volatile_Strong_Bullish",
    MarketRegime.NORMAL_VOLATILE_STRONG_BULLISH: "Normal_Volatile_Strong_Bullish",
    MarketRegime.LOW_VOLATILE_STRONG_BULLISH: "Low_Volatile_Strong_Bullish",
    MarketRegime.HIGH_VOLATILE_MILD_BULLISH: "High_Volatile_Mild_Bullish",
    MarketRegime.NORMAL_VOLATILE_MILD_BULLISH: "Normal_Volatile_Mild_Bullish",
    MarketRegime.LOW_VOLATILE_MILD_BULLISH: "Low_Volatile_Mild_Bullish",
    MarketRegime.HIGH_VOLATILE_NEUTRAL: "High_Volatile_Neutral",
    MarketRegime.NORMAL_VOLATILE_NEUTRAL: "Normal_Volatile_Neutral",
    MarketRegime.LOW_VOLATILE_NEUTRAL: "Low_Volatile_Neutral",
    MarketRegime.HIGH_VOLATILE_MILD_BEARISH: "High_Volatile_Mild_Bearish",
    MarketRegime.NORMAL_VOLATILE_MILD_BEARISH: "Normal_Volatile_Mild_Bearish",
    MarketRegime.LOW_VOLATILE_MILD_BEARISH: "Low_Volatile_Mild_Bearish",
    MarketRegime.HIGH_VOLATILE_STRONG_BEARISH: "High_Volatile_Strong_Bearish",
    MarketRegime.NORMAL_VOLATILE_STRONG_BEARISH: "Normal_Volatile_Strong_Bearish",
    MarketRegime.LOW_VOLATILE_STRONG_BEARISH: "Low_Volatile_Strong_Bearish",
    MarketRegime.BULLISH_TO_BEARISH_TRANSITION: "Bullish_To_Bearish_Transition",
    MarketRegime.BEARISH_TO_BULLISH_TRANSITION: "Bearish_To_Bullish_Transition",
    MarketRegime.VOLATILITY_EXPANSION: "Volatility_Expansion"
}

# Map old names (including typos) to standardized names
OLD_TO_STANDARD = {
    # Fix "voltatile" typo
    "high_voltatile_strong_bullish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_STRONG_BULLISH],
    "normal_voltatile_strong_bullish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_STRONG_BULLISH],
    "low_voltatile_strong_bullish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_STRONG_BULLISH],
    "high_voltatile_mild_bullish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_MILD_BULLISH],
    "normal_voltatile_mild_bullish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_MILD_BULLISH],
    "low_voltatile_mild_bullish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_MILD_BULLISH],
    "high_voltatile_neutral": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_NEUTRAL],
    "normal_voltatile_neutral": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_NEUTRAL],
    "low_voltatile_neutral": REGIME_NAMES[MarketRegime.LOW_VOLATILE_NEUTRAL],
    "high_voltatile_mild_bearish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_MILD_BEARISH],
    "normal_voltatile_mild_bearish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_MILD_BEARISH],
    "low_voltatile_mild_bearish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_MILD_BEARISH],
    "high_voltatile_strong_bearish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_STRONG_BEARISH],
    "normal_voltatile_strong_bearish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_STRONG_BEARISH],
    "low_voltatile_strong_bearish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_STRONG_BEARISH],
    
    # Map other naming conventions
    "high_volatile_strong_bullish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_STRONG_BULLISH],
    "normal_volatile_strong_bullish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_STRONG_BULLISH],
    "low_volatile_strong_bullish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_STRONG_BULLISH],
    "high_volatile_mild_bullish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_MILD_BULLISH],
    "normal_volatile_mild_bullish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_MILD_BULLISH],
    "low_volatile_mild_bullish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_MILD_BULLISH],
    "high_volatile_neutral": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_NEUTRAL],
    "normal_volatile_neutral": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_NEUTRAL],
    "low_volatile_neutral": REGIME_NAMES[MarketRegime.LOW_VOLATILE_NEUTRAL],
    "high_volatile_mild_bearish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_MILD_BEARISH],
    "normal_volatile_mild_bearish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_MILD_BEARISH],
    "low_volatile_mild_bearish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_MILD_BEARISH],
    "high_volatile_strong_bearish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_STRONG_BEARISH],
    "normal_volatile_strong_bearish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_STRONG_BEARISH],
    "low_volatile_strong_bearish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_STRONG_BEARISH],
    
    # Map lowercase versions
    "high volatile strong bullish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_STRONG_BULLISH],
    "normal volatile strong bullish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_STRONG_BULLISH],
    "low volatile strong bullish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_STRONG_BULLISH],
    "high volatile mild bullish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_MILD_BULLISH],
    "normal volatile mild bullish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_MILD_BULLISH],
    "low volatile mild bullish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_MILD_BULLISH],
    "high volatile neutral": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_NEUTRAL],
    "normal volatile neutral": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_NEUTRAL],
    "low volatile neutral": REGIME_NAMES[MarketRegime.LOW_VOLATILE_NEUTRAL],
    "high volatile mild bearish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_MILD_BEARISH],
    "normal volatile mild bearish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_MILD_BEARISH],
    "low volatile mild bearish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_MILD_BEARISH],
    "high volatile strong bearish": REGIME_NAMES[MarketRegime.HIGH_VOLATILE_STRONG_BEARISH],
    "normal volatile strong bearish": REGIME_NAMES[MarketRegime.NORMAL_VOLATILE_STRONG_BEARISH],
    "low volatile strong bearish": REGIME_NAMES[MarketRegime.LOW_VOLATILE_STRONG_BEARISH],
    
    # Map transitional regimes
    "bullish_to_bearish": REGIME_NAMES[MarketRegime.BULLISH_TO_BEARISH_TRANSITION],
    "bearish_to_bullish": REGIME_NAMES[MarketRegime.BEARISH_TO_BULLISH_TRANSITION],
    "volatility_expansion": REGIME_NAMES[MarketRegime.VOLATILITY_EXPANSION],
    "bullish to bearish": REGIME_NAMES[MarketRegime.BULLISH_TO_BEARISH_TRANSITION],
    "bearish to bullish": REGIME_NAMES[MarketRegime.BEARISH_TO_BULLISH_TRANSITION],
    "volatility expansion": REGIME_NAMES[MarketRegime.VOLATILITY_EXPANSION]
}

def standardize_regime_name(regime_name):
    """
    Convert any regime name to the standardized format.
    
    Args:
        regime_name (str): Input regime name, possibly with typos or inconsistent format
        
    Returns:
        str: Standardized regime name
    """
    if not regime_name:
        return None
    
    # Convert to lowercase for case-insensitive matching
    regime_lower = regime_name.lower()
    
    # Check if it's in the mapping
    if regime_lower in OLD_TO_STANDARD:
        return OLD_TO_STANDARD[regime_lower]
    
    # Check if it's already a standard name (case-insensitive)
    for standard_name in REGIME_NAMES.values():
        if regime_lower == standard_name.lower():
            return standard_name
    
    # If not found, log warning and return original
    logger.warning(f"Unknown market regime name: {regime_name}")
    return regime_name

def is_valid_regime_name(regime_name):
    """
    Check if a regime name is valid.
    
    Args:
        regime_name (str): Regime name to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not regime_name:
        return False
    
    # Convert to lowercase for case-insensitive matching
    regime_lower = regime_name.lower()
    
    # Check if it's in the mapping
    if regime_lower in OLD_TO_STANDARD:
        return True
    
    # Check if it's already a standard name (case-insensitive)
    for standard_name in REGIME_NAMES.values():
        if regime_lower == standard_name.lower():
            return True
    
    return False

def get_regime_enum(regime_name):
    """
    Get the enum value for a regime name.
    
    Args:
        regime_name (str): Regime name
        
    Returns:
        MarketRegime: Enum value for the regime
    """
    # Standardize the name first
    standard_name = standardize_regime_name(regime_name)
    
    # Find the enum value
    for enum_value, name in REGIME_NAMES.items():
        if name == standard_name:
            return enum_value
    
    return None

def get_all_regime_names():
    """
    Get all valid regime names.
    
    Returns:
        list: List of all valid regime names
    """
    return list(REGIME_NAMES.values())

def get_directional_component(regime_name):
    """
    Extract the directional component from a regime name.
    
    Args:
        regime_name (str): Regime name
        
    Returns:
        str: Directional component (Strong_Bullish, Mild_Bullish, Neutral, Mild_Bearish, Strong_Bearish)
    """
    # Standardize the name first
    standard_name = standardize_regime_name(regime_name)
    
    if not standard_name:
        return None
    
    # Extract directional component
    if "Strong_Bullish" in standard_name:
        return "Strong_Bullish"
    elif "Mild_Bullish" in standard_name:
        return "Mild_Bullish"
    elif "Strong_Bearish" in standard_name:
        return "Strong_Bearish"
    elif "Mild_Bearish" in standard_name:
        return "Mild_Bearish"
    elif "Neutral" in standard_name:
        return "Neutral"
    elif standard_name == REGIME_NAMES[MarketRegime.BULLISH_TO_BEARISH_TRANSITION]:
        return "Bullish_To_Bearish"
    elif standard_name == REGIME_NAMES[MarketRegime.BEARISH_TO_BULLISH_TRANSITION]:
        return "Bearish_To_Bullish"
    elif standard_name == REGIME_NAMES[MarketRegime.VOLATILITY_EXPANSION]:
        return "Volatility_Expansion"
    
    return None

def get_volatility_component(regime_name):
    """
    Extract the volatility component from a regime name.
    
    Args:
        regime_name (str): Regime name
        
    Returns:
        str: Volatility component (High_Volatile, Normal_Volatile, Low_Volatile)
    """
    # Standardize the name first
    standard_name = standardize_regime_name(regime_name)
    
    if not standard_name:
        return None
    
    # Extract volatility component
    if "High_Volatile" in standard_name:
        return "High_Volatile"
    elif "Normal_Volatile" in standard_name:
        return "Normal_Volatile"
    elif "Low_Volatile" in standard_name:
        return "Low_Volatile"
    
    return None

def fix_regime_names_in_dataframe(df, column_name):
    """
    Fix regime names in a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame to fix
        column_name (str): Column name containing regime names
        
    Returns:
        pd.DataFrame: DataFrame with fixed regime names
    """
    if column_name not in df.columns:
        logger.warning(f"Column {column_name} not found in DataFrame")
        return df
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Fix regime names
    df_copy[column_name] = df_copy[column_name].apply(standardize_regime_name)
    
    return df_copy
