import logging
import logging.config
import yaml
import os
from pathlib import Path
from functools import lru_cache

@lru_cache()
def setup_logging(
    default_path=None,
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration"""
    if default_path is None:
        # Default to src/config/logging.yaml relative to this file
        # src/utils/logger.py -> src/utils/../config/logging.yaml -> src/config/logging.yaml
        default_path = Path(__file__).resolve().parent.parent / 'config' / 'logging.yaml'

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
        
    config_path = Path(path)
    
    if config_path.exists():
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error loading logging config: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)

def get_logger(name: str):
    """Get a logger instance with the specified name"""
    # Ensure logging is set up
    setup_logging()
    return logging.getLogger(name)
