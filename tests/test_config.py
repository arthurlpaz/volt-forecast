"""Tests for the typed configuration layer.

These exercise the real YAML files in `configs/`, not fixtures. That is
deliberate: the most likely way this layer breaks is someone editing a YAML
file into a shape the models no longer accept, and a test against a fixture
would happily pass while the actual platform fails to boot.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from energycast.config import Settings, get_settings
from energycast.config.settings import SplitConfig, _load_yaml


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Stop a cached Settings from leaking between tests that patch env/YAML."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestRealConfigFiles:
    def test_real_yaml_files_load_and_validate(self):
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert settings.base.environment in {"development", "staging", "production"}
        assert settings.data.source.target_column
        assert settings.model.sequence.sequence_length > 0

    def test_settings_are_cached_per_process(self):
        assert get_settings() is get_settings()

    def test_missing_config_file_raises_readable_error(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_yaml("does_not_exist.yaml")


class TestSplitValidation:
    def test_valid_ratios_are_accepted(self):
        split = SplitConfig(train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)
        assert split.train_ratio == 0.7

    @pytest.mark.parametrize(
        ("train", "validation", "test"),
        [
            (0.7, 0.15, 0.10),  # sums to 0.95 — silently discards 5% of the data
            (0.7, 0.20, 0.20),  # sums to 1.10 — splits would overlap and leak
        ],
    )
    def test_ratios_not_summing_to_one_are_rejected(self, train, validation, test):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            SplitConfig(train_ratio=train, validation_ratio=validation, test_ratio=test)

    def test_float_representation_error_is_tolerated(self):
        # This split is correct, but does not sum to exactly 1.0 in binary
        # floating point (0.9999999999999999). A strict `!= 1.0` check would
        # reject it, so the validator must compare with a tolerance.
        assert (0.7 + 0.2 + 0.1) != 1.0
        SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)

    def test_zero_and_negative_ratios_are_rejected(self):
        with pytest.raises(ValidationError):
            SplitConfig(train_ratio=1.2, validation_ratio=-0.1, test_ratio=-0.1)


class TestEnvironmentOverrides:
    """The module docstring and README both promise ENERGYCAST_-prefixed env
    overrides. This is the contract that milestone 8 (FastAPI in production,
    pointing at a real MLflow backend) will depend on, so it needs to hold.
    """

    def test_env_var_overrides_yaml_value(self, monkeypatch):
        monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", "postgresql://override")
        get_settings.cache_clear()

        assert get_settings().base.mlflow.tracking_uri == "postgresql://override"

    def test_unset_fields_still_come_from_yaml(self, monkeypatch):
        monkeypatch.setenv("ENERGYCAST_BASE__MLFLOW__TRACKING_URI", "postgresql://override")
        get_settings.cache_clear()

        settings = get_settings()
        # The override must not blow away its siblings in the same nested model.
        assert settings.base.mlflow.experiment_name == "energycast-ai"
        assert settings.base.environment == "development"
