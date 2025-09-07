# src/utils/config_parser.py
import yaml
from typing import Dict, Any, Optional
import os

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (str, optional): Path to the YAML config file. If None, defaults
            to 'configs/config.yaml' relative to the project root.

    Returns:
        dict: Parsed configuration

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML is invalid
    """
    # Determine default path when not provided
    if config_path is None:
        # Allow override via environment variable
        config_path = os.environ.get('CONFIG_PATH', os.path.join('configs', 'config.yaml'))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config
