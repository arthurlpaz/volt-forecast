"""Structured logging for the EnergyCast AI platform.

Design rationale
-----------------
Every module in the platform logs through `get_logger(__name__)` rather than
calling `logging.getLogger` directly. That indirection is the whole point:
the choice of *format* (JSON vs. plain text), *level*, and *destination* is
made once, here, driven by `configs/base.yaml`. Switching the entire platform
from human-readable text to machine-parseable JSON is a config flag, not a
migration across every call site.

JSON is the default because the logs from milestones 9-11 (prediction
logging, drift detection, retraining decisions) are meant to be queried, not
read by eye. A drift event that can be filtered with
`jq 'select(.event == "drift_detected")'` is worth more than one that needs a
regex.

Handlers are attached to the `energycast` package logger, not to the root
logger. Configuring root would mean fighting with — and reformatting the
output of — every third-party library in the dependency tree (mlflow,
uvicorn, torch). Scoping to our own namespace keeps that blast radius at
zero.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

PACKAGE_LOGGER_NAME = "energycast"

# Included so that JSON records carry the context needed to trace a log line
# back to its origin without opening the source.
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

    Arguments override `configs/base.yaml`; when omitted, settings are read
    from there. They are injectable so that tests (and any future CLI `-v`
    flag) can drive logging without mutating YAML on disk.

    Idempotent by default: repeated calls are a no-op, because module-level
    `get_logger` calls across the package would otherwise stack a duplicate
    handler on every import and emit each record N times. Pass `force=True`
    to deliberately reconfigure.
    """
    global _configured
    if _configured and not force:
        return

    if level is None or json_format is None:
        # Imported lazily: config imports nothing from utils, and keeping this
        # import inside the function means a caller who passes both arguments
        # explicitly (i.e. tests) never needs valid YAML on disk at all.
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

    # Records stop here rather than bubbling to root, where a library's
    # basicConfig() could reformat or duplicate them.
    logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name`, configuring the package logger on first use.

    Call as `get_logger(__name__)`. Names outside the `energycast` namespace
    are re-parented under it, so that a logger obtained from a script or a
    notebook still inherits the platform's handler and level instead of
    silently falling through to root.
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
