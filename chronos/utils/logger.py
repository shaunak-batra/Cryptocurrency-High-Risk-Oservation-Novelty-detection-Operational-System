"""
Structured Logging for CHRONOS

Provides JSON-formatted structured logs for production monitoring.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger wrapper with context support."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """
        Initialize structured logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : str
            Log level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Add JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        
        self.context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """Bind context to logger."""
        self.context.update(kwargs)
        return self
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context."""
        extra = {**self.context, **kwargs}
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            None
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra={"extra_fields": {**self.context, **kwargs}})


def setup_logging(level: str = "INFO", json_format: bool = True):
    """
    Configure application logging.
    
    Parameters
    ----------
    level : str
        Log level
    json_format : bool
        Whether to use JSON formatting
    """
    import os
    level = os.getenv("LOG_LEVEL", level)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    root_logger.addHandler(handler)
    
    return root_logger


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Parameters
    ----------
    name : str
        Logger name
    level : str
        Log level
        
    Returns
    -------
    StructuredLogger
        Logger instance
    """
    return StructuredLogger(name, level)


# Default logger
logger = get_logger("chronos")
