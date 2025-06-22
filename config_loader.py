import yaml
from pathlib import Path
from datetime import datetime

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config file with error handling"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            validate_config(config)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config: {str(e)}")

def validate_config(config: dict) -> None:
    """Validate config structure"""
    required_sections = ['paths', 'model', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

def get_results_dir(config: dict) -> Path:
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config['paths']['results_base']) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
