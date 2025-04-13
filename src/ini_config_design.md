# INI-Based Configuration System Design

## Overview

This document outlines the design for converting the existing YAML-based configuration system to an INI-based system while preserving all functionality and configuration options.

## Challenges

1. **Nested Structure Conversion**: INI files don't natively support nested structures like YAML, so we need to flatten the hierarchy.
2. **List/Array Representation**: INI doesn't have a standard way to represent lists or arrays.
3. **Type Preservation**: We need to ensure proper type conversion (boolean, integer, float, string).
4. **Backward Compatibility**: The new system should work with minimal changes to existing code.

## Design Approach

### 1. Flattening Hierarchical Structure

We'll use section names and key prefixes to represent the hierarchy:

**YAML Structure:**
```yaml
technical_indicators:
  ema_indicators:
    enabled: true
    periods: [20, 100, 200]
```

**INI Structure:**
```ini
[technical_indicators.ema_indicators]
enabled = true
periods = 20,100,200
```

### 2. List/Array Representation

We'll use comma-separated values for simple lists and indexed keys for complex objects:

**Simple Lists:**
```ini
[technical_indicators.ema_indicators]
periods = 20,100,200
timeframes = 5m,10m,15m
```

**Complex Objects in Lists:**
```ini
[data_processing.timeframes.0]
name = 5m
minutes = 5
lookback = 60

[data_processing.timeframes.1]
name = 10m
minutes = 10
lookback = 30
```

### 3. Dictionary Representation

For dictionaries/mappings, we'll use dot notation in the keys:

**YAML:**
```yaml
iv_indicators:
  dte_lookback_mapping:
    "0-7": 30
    "8-14": 60
```

**INI:**
```ini
[technical_indicators.iv_indicators]
dte_lookback_mapping.0-7 = 30
dte_lookback_mapping.8-14 = 60
```

### 4. Type Handling

We'll implement type conversion in the ConfigManager:

```ini
[technical_indicators.ema_indicators]
enabled = true  # Boolean
periods = 20,100,200  # Integer list
use_slope = true  # Boolean
```

### 5. Section Organization

The INI file will be organized into these main sections:

```ini
[technical_indicators.ema_indicators]
# EMA indicator settings

[technical_indicators.atr_indicators]
# ATR indicator settings

[technical_indicators.iv_indicators]
# IV indicator settings

[technical_indicators.premium_indicators]
# Premium indicator settings

[technical_indicators.trending_oi_pa]
# Trending OI with PA settings

[technical_indicators.greek_sentiment]
# Greek sentiment settings

[technical_indicators.vwap_indicators]
# VWAP indicator settings

[technical_indicators.volume_indicators]
# Volume indicator settings

[market_regime.directional_regimes.strong_bullish]
# Strong bullish regime thresholds

[market_regime.directional_regimes.bullish]
# Bullish regime thresholds

# ... other directional regimes ...

[market_regime.volatility_regimes.very_low_vol]
# Very low volatility regime thresholds

# ... other volatility regimes ...

[market_regime.liquidity_regimes.high_liquidity]
# High liquidity regime thresholds

# ... other liquidity regimes ...

[market_regime.composite_regimes.high_volatility]
# High volatility composite regime mappings

[market_regime.composite_regimes.low_volatility]
# Low volatility composite regime mappings

[dynamic_weighting.default_weights]
# Default component weights

[dynamic_weighting.adjustment]
# Dynamic adjustment settings

[dynamic_weighting.performance_metrics]
# Performance metrics for weight adjustment

[confidence_score]
# Confidence score settings

[confidence_score.calculation]
# Confidence score calculation weights

[data_processing.symbols.NIFTY]
# NIFTY-specific settings

[data_processing.dte_buckets]
# DTE bucketing settings

# ... timeframes and filtering settings ...
```

## Implementation Plan

### 1. ConfigManager Updates

The ConfigManager class will be updated to:

1. Load INI files instead of YAML
2. Convert flattened INI structure back to nested dictionary
3. Handle type conversion
4. Maintain the same API for accessing configuration

```python
import configparser
import os
import logging
from typing import Dict, Any, Optional, List, Union

class ConfigManager:
    def __init__(self, config_path: str = None):
        # Initialize with default or specified config path
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'default_config.ini'
        )
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from INI file and convert to nested dictionary."""
        try:
            config_parser = configparser.ConfigParser()
            config_parser.read(self.config_path)
            
            # Convert flat INI structure to nested dictionary
            self.config = self._build_nested_dict(config_parser)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            self._validate_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _build_nested_dict(self, config_parser: configparser.ConfigParser) -> Dict[str, Any]:
        """Build nested dictionary from flat INI structure."""
        result = {}
        
        for section in config_parser.sections():
            section_parts = section.split('.')
            current = result
            
            # Navigate to the correct nested level
            for i, part in enumerate(section_parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add the section's key-value pairs
            last_part = section_parts[-1]
            if last_part not in current:
                current[last_part] = {}
            
            for key, value in config_parser[section].items():
                # Handle nested keys (e.g., "dte_lookback_mapping.0-7")
                if '.' in key:
                    key_parts = key.split('.')
                    nested_current = current[last_part]
                    
                    for nested_key in key_parts[:-1]:
                        if nested_key not in nested_current:
                            nested_current[nested_key] = {}
                        nested_current = nested_current[nested_key]
                    
                    nested_current[key_parts[-1]] = self._convert_value(value)
                else:
                    current[last_part][key] = self._convert_value(value)
        
        return result
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Handle boolean
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        if value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Handle lists
        if ',' in value:
            items = [item.strip() for item in value.split(',')]
            
            # Try to convert to numeric if possible
            try:
                return [int(item) for item in items]
            except ValueError:
                try:
                    return [float(item) for item in items]
                except ValueError:
                    return items
        
        # Handle numeric values
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value  # Keep as string
    
    # Rest of the methods remain largely unchanged
    def _validate_config(self) -> None:
        """Validate the configuration structure and required fields."""
        # Implementation remains the same
        pass
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save the current configuration to an INI file."""
        save_path = config_path or self.config_path
        try:
            config_parser = configparser.ConfigParser()
            
            # Convert nested dictionary to flat INI structure
            self._build_flat_config(config_parser, self.config)
            
            with open(save_path, 'w') as file:
                config_parser.write(file)
            
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def _build_flat_config(self, config_parser: configparser.ConfigParser, 
                          config_dict: Dict[str, Any], prefix: str = '') -> None:
        """Build flat INI structure from nested dictionary."""
        for key, value in config_dict.items():
            section = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                self._build_flat_config(config_parser, value, section)
            else:
                # Add key-value pair to the appropriate section
                if section not in config_parser:
                    config_parser[section] = {}
                
                # Convert value to string representation
                if isinstance(value, bool):
                    config_parser[section][key] = str(value).lower()
                elif isinstance(value, (list, tuple)):
                    config_parser[section][key] = ','.join(str(item) for item in value)
                else:
                    config_parser[section][key] = str(value)
    
    # Accessor methods remain unchanged
    def get_config(self) -> Dict[str, Any]:
        return self.config
    
    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        indicators_config = self.config.get('technical_indicators', {})
        if indicator_name not in indicators_config:
            self.logger.warning(f"Indicator {indicator_name} not found in configuration")
            return {}
        return indicators_config[indicator_name]
    
    def get_market_regime_config(self) -> Dict[str, Any]:
        return self.config.get('market_regime', {})
    
    def get_dynamic_weighting_config(self) -> Dict[str, Any]:
        return self.config.get('dynamic_weighting', {})
    
    def get_confidence_score_config(self) -> Dict[str, Any]:
        return self.config.get('confidence_score', {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        return self.config.get('data_processing', {})
    
    def update_config_section(self, section: str, new_config: Dict[str, Any]) -> None:
        if section not in self.config:
            self.logger.warning(f"Creating new configuration section: {section}")
        
        self.config[section] = new_config
        self.logger.info(f"Updated configuration section: {section}")
        
    def update_indicator_config(self, indicator_name: str, new_config: Dict[str, Any]) -> None:
        if 'technical_indicators' not in self.config:
            self.config['technical_indicators'] = {}
            
        self.config['technical_indicators'][indicator_name] = new_config
        self.logger.info(f"Updated configuration for indicator: {indicator_name}")
```

### 2. Default Configuration File

We'll create a default_config.ini file that contains all the same configuration options as the current YAML file, but in the flattened INI format.

### 3. Minimal Changes to Other Components

The IndicatorFactory and MarketRegimeClassifier classes should require minimal changes since they interact with the configuration through the ConfigManager's API, which will remain the same.

## Advantages of This Approach

1. **Backward Compatibility**: The nested dictionary structure returned by ConfigManager remains the same, so other components don't need significant changes.
2. **Readability**: The INI format is still readable and maintainable.
3. **Comprehensive**: All configuration options from the YAML file are preserved.
4. **Extensible**: The system can be easily extended to add new configuration options.

## Implementation Sequence

1. Implement the updated ConfigManager class
2. Create the default_config.ini file
3. Update any direct YAML references in other components
4. Test the configuration loading and access
5. Verify that all components work with the new configuration system
