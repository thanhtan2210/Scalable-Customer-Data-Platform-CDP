import logging.config
import os
from .config import DEBUG

def setup_logging():
    log_level = "DEBUG" if DEBUG else "INFO"
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": log_level,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
            "cdp": {  # Application specific logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)
