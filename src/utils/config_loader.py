"""
Configuration loader utility for Neural-Decoder.
Handles loading and accessing YAML configuration files.
"""

import yaml
import os
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration settings from YAML file."""
    
    def __init__(self, config_path="configs/config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        # Get the project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        full_config_path = project_root / self.config_path
        
        try:
            with open(full_config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {full_config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration if file loading fails."""
        return {
            'device': {
                'mac_address': 'f0:17:3b:41:ec:7d',
                'sampling_rate': 200,
                'channel_of_interest': 0
            },
            'signal_processing': {
                'window_size_sec': 3.0,
                'overlap_percent': 0.5,
                'alpha_band': {'low_freq': 8.0, 'high_freq': 12.0}
            },
            'ml': {
                'training': {'epochs': 20, 'batch_size': 32}
            }
        }
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to config value (e.g., 'device.mac_address')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_device_config(self):
        """Get device configuration section."""
        return self.config.get('device', {})
    
    def get_signal_processing_config(self):
        """Get signal processing configuration section."""
        return self.config.get('signal_processing', {})
    
    def get_ml_config(self):
        """Get machine learning configuration section."""
        return self.config.get('ml', {})
    
    def get_visualization_config(self):
        """Get visualization configuration section."""
        return self.config.get('visualization', {})


# Global configuration instance
config = ConfigLoader() 