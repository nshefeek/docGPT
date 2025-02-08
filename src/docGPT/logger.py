import structlog
import logging
import sys

def setup_logger():
    """
    Configures structlog for structured logging.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),  # Outputs logs in JSON format
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging for compatibility
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console logs
            logging.FileHandler("app.log", mode="a"),  # File logs
        ],
    )

    return structlog.get_logger()

logger = setup_logger()
