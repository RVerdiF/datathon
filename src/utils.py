"""
Utility functions for the Passos Mágicos ML project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import json


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        log_format: Log message format
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("passos_magicos")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics dictionary to JSON file."""
    ensure_dir(Path(filepath).parent)
    
    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_classification_report(report: Dict[str, Any]) -> str:
    """Format sklearn classification report dict as string."""
    lines = []
    lines.append(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("")
    
    for label in ['0', '1']:
        if label in report:
            r = report[label]
            lines.append(
                f"{label:>12} {r['precision']:>10.2f} {r['recall']:>10.2f} "
                f"{r['f1-score']:>10.2f} {r['support']:>10.0f}"
            )
    
    lines.append("")
    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {report['accuracy']:>10.2f} {report['macro avg']['support']:>10.0f}")
    lines.append(f"{'macro avg':>12} {report['macro avg']['precision']:>10.2f} "
                 f"{report['macro avg']['recall']:>10.2f} {report['macro avg']['f1-score']:>10.2f} "
                 f"{report['macro avg']['support']:>10.0f}")
    lines.append(f"{'weighted avg':>12} {report['weighted avg']['precision']:>10.2f} "
                 f"{report['weighted avg']['recall']:>10.2f} {report['weighted avg']['f1-score']:>10.2f} "
                 f"{report['weighted avg']['support']:>10.0f}")
    
    return '\n'.join(lines)
