"""Configuration management for the transaction anomaly detection system."""

import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file and override with environment variables.
    
    Environment variables take precedence over YAML values.
    Format: Use nested keys with underscores, e.g., REAL_TIME_KAFKA_BOOTSTRAP_SERVERS
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively apply environment variable overrides to config.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for environment variable names
        
    Returns:
        Updated configuration dictionary
    """
    result = {}
    
    for key, value in config.items():
        env_key = f"{prefix}_{key}".upper() if prefix else key.upper()
        
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = _apply_env_overrides(value, env_key)
        else:
            # Check for environment variable override
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Try to convert to appropriate type
                if isinstance(value, bool):
                    result[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(value, int):
                    try:
                        result[key] = int(env_value)
                    except ValueError:
                        result[key] = value
                elif isinstance(value, float):
                    try:
                        result[key] = float(env_value)
                    except ValueError:
                        result[key] = value
                else:
                    result[key] = env_value
            else:
                result[key] = value
    
    return result


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
