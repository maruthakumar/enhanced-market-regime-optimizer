import configparser
import os
import logging
from typing import Dict, Any, Optional, List, Union

class ConfigManager:
    """
    Configuration manager for market regime formation.
    Handles loading, validating, and accessing configuration settings from INI files.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
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
    
    def _validate_config(self) -> None:
        """Validate the configuration structure and required fields."""
        required_sections = [
            'technical_indicators', 
            'market_regime', 
            'dynamic_weighting',
            'confidence_score',
            'data_processing'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate technical indicators
        if not any(self.config['technical_indicators'].get(indicator, {}).get('enabled', False) 
                  for indicator in self.config['technical_indicators']):
            raise ValueError("At least one technical indicator must be enabled")
            
        # Validate market regime thresholds
        self._validate_regime_thresholds()
        
        # Validate dynamic weighting
        weights = self.config['dynamic_weighting']['default_weights']
        if abs(sum(weights.values()) - 1.0) > 0.001:
            self.logger.warning("Default weights do not sum to 1.0, they will be normalized")
            total = sum(weights.values())
            for key in weights:
                weights[key] = weights[key] / total
                
        self.logger.info("Configuration validation successful")
    
    def _validate_regime_thresholds(self) -> None:
        """Validate market regime thresholds for consistency."""
        # Check directional regimes
        dir_regimes = self.config['market_regime']['directional_regimes']
        for regime, thresholds in dir_regimes.items():
            if thresholds['min_threshold'] >= thresholds['max_threshold']:
                raise ValueError(f"Invalid thresholds for directional regime {regime}")
        
        # Check volatility regimes
        vol_regimes = self.config['market_regime']['volatility_regimes']
        for regime, thresholds in vol_regimes.items():
            if thresholds['min_percentile'] >= thresholds['max_percentile']:
                raise ValueError(f"Invalid percentiles for volatility regime {regime}")
        
        # Check liquidity regimes
        liq_regimes = self.config['market_regime']['liquidity_regimes']
        for regime, thresholds in liq_regimes.items():
            if thresholds['min_percentile'] >= thresholds['max_percentile']:
                raise ValueError(f"Invalid percentiles for liquidity regime {regime}")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save the current configuration to an INI file.
        
        Args:
            config_path: Path to save the configuration. If None, uses the current config path.
        """
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
            current_prefix = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                self._build_flat_config(config_parser, value, current_prefix)
            else:
                # Add key-value pair to the appropriate section
                section = prefix if prefix else 'DEFAULT'
                if section not in config_parser:
                    config_parser[section] = {}
                
                # Convert value to string representation
                if isinstance(value, bool):
                    config_parser[section][key] = str(value).lower()
                elif isinstance(value, (list, tuple)):
                    config_parser[section][key] = ','.join(str(item) for item in value)
                else:
                    config_parser[section][key] = str(value)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config
    
    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific technical indicator.
        
        Args:
            indicator_name: Name of the indicator (e.g., 'ema_indicators', 'greek_sentiment')
            
        Returns:
            Configuration dictionary for the specified indicator
        """
        indicators_config = self.config.get('technical_indicators', {})
        if indicator_name not in indicators_config:
            self.logger.warning(f"Indicator {indicator_name} not found in configuration")
            return {}
        return indicators_config[indicator_name]
    
    def get_market_regime_config(self) -> Dict[str, Any]:
        """Get market regime configuration."""
        return self.config.get('market_regime', {})
    
    def get_dynamic_weighting_config(self) -> Dict[str, Any]:
        """Get dynamic weighting configuration."""
        return self.config.get('dynamic_weighting', {})
    
    def get_confidence_score_config(self) -> Dict[str, Any]:
        """Get confidence score configuration."""
        return self.config.get('confidence_score', {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.config.get('data_processing', {})
    
    def update_config_section(self, section: str, new_config: Dict[str, Any]) -> None:
        """
        Update a section of the configuration.
        
        Args:
            section: Section name to update
            new_config: New configuration dictionary for the section
        """
        if section not in self.config:
            self.logger.warning(f"Creating new configuration section: {section}")
        
        self.config[section] = new_config
        self.logger.info(f"Updated configuration section: {section}")
        
    def update_indicator_config(self, indicator_name: str, new_config: Dict[str, Any]) -> None:
        """
        Update configuration for a specific technical indicator.
        
        Args:
            indicator_name: Name of the indicator to update
            new_config: New configuration dictionary for the indicator
        """
        if 'technical_indicators' not in self.config:
            self.config['technical_indicators'] = {}
            
        self.config['technical_indicators'][indicator_name] = new_config
        self.logger.info(f"Updated configuration for indicator: {indicator_name}")
