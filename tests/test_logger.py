"""Tests for the structured logging layer.

Every future module logs through `get_logger`, so the guarantees asserted
here — one handler, parseable JSON, correct namespace — are load-bearing for
the whole platform.
"""

from __future__ import annotations

import json
import logging

import pytest

from energycast.utils import configure_logging, get_logger, reset_logging
from energycast.utils.logger import PACKAGE_LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset():
    reset_logging()
    yield
    reset_logging()


class TestGetLogger:
    def test_returns_logger_in_package_namespace(self):
        assert get_logger("energycast.data.loader").name == "energycast.data.loader"

    def test_foreign_names_are_reparented_under_the_package(self):
        # A logger from a script or notebook must still inherit our handler
        # rather than falling through to an unconfigured root logger.
        assert get_logger("__main__").name == "energycast.__main__"

    def test_package_root_name_is_not_doubled(self):
        assert get_logger("energycast").name == "energycast"


class TestConfiguration:
    def test_json_output_is_machine_parseable(self, capsys):
        configure_logging(level="INFO", json_format=True, force=True)
        get_logger("energycast.test").info("drift detected", extra={"event": "drift_detected"})

        record = json.loads(capsys.readouterr().out.strip())
        assert record["message"] == "drift detected"
        assert record["levelname"] == "INFO"
        assert record["name"] == "energycast.test"
        # `extra` fields must survive into the JSON, since milestones 9-11
        # rely on filtering logs by structured keys.
        assert record["event"] == "drift_detected"

    def test_text_format_is_not_json(self, capsys):
        configure_logging(level="INFO", json_format=False, force=True)
        get_logger("energycast.test").info("hello")

        out = capsys.readouterr().out
        assert "hello" in out
        with pytest.raises(json.JSONDecodeError):
            json.loads(out.strip())

    def test_level_is_respected(self, capsys):
        configure_logging(level="WARNING", json_format=True, force=True)
        logger = get_logger("energycast.test")
        logger.debug("suppressed")
        logger.info("suppressed")
        logger.warning("emitted")

        lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
        assert len(lines) == 1
        assert json.loads(lines[0])["message"] == "emitted"

    def test_repeated_calls_do_not_stack_handlers(self, capsys):
        # Regression guard: every module calls get_logger at import time. If
        # configure_logging were not idempotent, a record would be emitted
        # once per imported module.
        for _ in range(5):
            configure_logging(level="INFO", json_format=True)
        get_logger("energycast.test").info("once")

        lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
        assert len(lines) == 1

    def test_does_not_propagate_to_root(self):
        configure_logging(level="INFO", json_format=True, force=True)
        assert logging.getLogger(PACKAGE_LOGGER_NAME).propagate is False

    def test_falls_back_to_yaml_when_no_arguments_given(self):
        configure_logging(force=True)
        # configs/base.yaml declares INFO.
        assert logging.getLogger(PACKAGE_LOGGER_NAME).level == logging.INFO
