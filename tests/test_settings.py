"""Tests for AEGIS-AD configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aegis_ad.settings import (
    ConfigurationError,
    Settings,
    get_project_root,
    load_settings,
    validate_settings,
)


def test_default_config_loads_successfully():
    """Verify that the default configuration loads without errors."""
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.app.get("name") == "AEGIS-AD Preventive AI Safety System"
    assert settings.app.get("env") == "development"
    assert settings.logging.get("level") == "INFO"
    assert "data_dir" in settings.paths


def test_missing_config_file_raises_clear_error(tmp_path: Path):
    """Verify that specifying a non-existent config file raises ConfigurationError with path."""
    non_existent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(non_existent)
    assert "Configuration file not found" in str(exc_info.value)
    assert str(non_existent) in str(exc_info.value)


def test_invalid_yaml_syntax_raises_clear_error(tmp_path: Path):
    """Verify that malformed YAML produces a clear ConfigurationError."""
    bad_yaml = tmp_path / "invalid.yaml"
    bad_yaml.write_text("app: [unclosed list", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(bad_yaml)
    assert "Failed to read or parse YAML" in str(exc_info.value)


def test_invalid_environment_validation_error():
    """Verify that an invalid app environment fails validation."""
    raw = {
        "app": {"name": "Test App", "env": "unsupported_env"},
        "logging": {"level": "INFO"},
    }
    errors = validate_settings(raw)
    assert any("app.env must be one of" in err for err in errors)


def test_invalid_log_level_validation_error():
    """Verify that an invalid logging level fails validation."""
    raw = {
        "app": {"name": "Test App", "env": "development"},
        "logging": {"level": "NOT_A_LEVEL"},
    }
    errors = validate_settings(raw)
    assert any("logging.level must be one of" in err for err in errors)


def test_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch):
    """Verify that AEGIS_ prefixed environment variables override config values."""
    monkeypatch.setenv("AEGIS_APP__ENV", "production")
    monkeypatch.setenv("AEGIS_LOGGING__LEVEL", "DEBUG")

    settings = load_settings()
    assert settings.app.get("env") == "production"
    assert settings.logging.get("level") == "DEBUG"
