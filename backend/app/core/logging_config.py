import logging
import json
import os
from datetime import datetime


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    is_production = os.getenv("ENV", "development") == "production"

    handler = logging.StreamHandler()
    if is_production:
        handler.setFormatter(JSONFormatter())
    else:
        # Development: human-readable
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    # Use force=True to ensure basicConfig overrides any existing handlers
    logging.basicConfig(
        level=getattr(logging, log_level), handlers=[handler], force=True
    )

    # Không log sensitive data
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
