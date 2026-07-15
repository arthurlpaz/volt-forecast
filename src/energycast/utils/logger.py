"""Structured logging.

Every module logs through `get_logger(__name__)`, so format and level are
decided once, here, from configs/base.yaml.

Handlers attach to the `energycast` logger rather than root, to avoid
reformatting the output of every library in the dependency tree.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

PACKAGE_LOGGER_NAME = "energycast"

_JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging(
    level: str | None = None,
    json_format: bool | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure the `energycast` package logger.

    Arguments override configs/base.yaml; omitted ones are read from it.

    Idempotent by default, since module-level `get_logger` calls would
    otherwise stack a handler per import and emit each record N times.
    """
    global _configured
    if _configured and not force:
        return

    if level is None or json_format is None:
        # Lazy, so callers passing both arguments need no YAML on disk.
        from energycast.config import get_settings

        log_cfg = get_settings().base.logging
        level = level if level is not None else log_cfg.level
        json_format = json_format if json_format is not None else log_cfg.json_format

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(jsonlogger.JsonFormatter(_JSON_FORMAT))
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    logger.addHandler(handler)

    # Stop here rather than bubbling to root, where a library's basicConfig()
    # could reformat or duplicate these records.
    logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name`, configuring the package logger on first use.

    Names outside the `energycast` namespace are re-parented under it, so a
    logger from a script or notebook still inherits our handler and level.
    """
    configure_logging()

    if name == PACKAGE_LOGGER_NAME or name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def reset_logging() -> None:
    """Drop handlers and clear the configured flag. Intended for test teardown."""
    global _configured
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    _configured = False
