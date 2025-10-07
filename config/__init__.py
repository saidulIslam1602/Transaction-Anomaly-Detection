"""Configuration management for the transaction anomaly detection system."""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_env_variable(var_name: str, default: Any = None) -> Any:
    """
    Get environment variable with fallback.
    
    Args:
        var_name: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(var_name, default)
