"""Configuration loader and validation for AEGIS-AD."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""


VALID_ENVIRONMENTS = {"development", "staging", "production", "testing"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class SectionConfig:
    """Attribute-style wrapper around a dictionary section."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Configuration key '{name}' not found") from None

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class Settings:
    """Application settings root."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self.app = SectionConfig(raw.get("app", {}))
        self.logging = SectionConfig(raw.get("logging", {}))
        self.paths = SectionConfig(raw.get("paths", {}))

    @property
    def raw(self) -> dict[str, Any]:
        """Return the raw underlying configuration dictionary."""
        return self._raw


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def _convert_value(val: str) -> Any:
    """Convert an environment variable string into a native Python value."""
    lowered = val.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Apply environment variables prefixed with AEGIS_ (e.g. AEGIS_APP__ENV=production)."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("AEGIS_"):
            continue
        parts = env_key[6:].lower().split("__")
        if len(parts) >= 2:
            section = parts[0]
            key = "_".join(parts[1:])
            data.setdefault(section, {})[key] = _convert_value(env_val)


def validate_settings(raw: dict[str, Any]) -> list[str]:
    """Validate raw configuration settings dictionary. Returns a list of error strings."""
    errors: list[str] = []

    if not isinstance(raw, dict):
        return ["Configuration root must be a dictionary."]

    app = raw.get("app")
    if not isinstance(app, dict):
        errors.append("Section 'app' is missing or must be a dictionary.")
    else:
        name = app.get("name")
        if not name or not isinstance(name, str):
            errors.append("app.name must be a non-empty string.")
        env = app.get("env", "development")
        if env not in VALID_ENVIRONMENTS:
            errors.append(f"app.env must be one of {sorted(VALID_ENVIRONMENTS)}, got: '{env}'")

    logging_cfg = raw.get("logging")
    if not isinstance(logging_cfg, dict):
        errors.append("Section 'logging' is missing or must be a dictionary.")
    else:
        level = logging_cfg.get("level", "INFO")
        if not isinstance(level, str) or level.upper() not in VALID_LOG_LEVELS:
            errors.append(f"logging.level must be one of {sorted(VALID_LOG_LEVELS)}, got: '{level}'")

    return errors


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load configuration from file, apply env overrides, validate, and return Settings."""
    load_dotenv()

    if config_path is None:
        target_path = get_project_root() / "config" / "default.yaml"
    else:
        target_path = Path(config_path)

    if not target_path.exists():
        raise ConfigurationError(f"Configuration file not found: {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigurationError(f"Failed to read or parse YAML config from {target_path}: {exc}") from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ConfigurationError(f"YAML configuration at {target_path} must define a mapping, got {type(raw).__name__}")

    _apply_env_overrides(raw)

    errors = validate_settings(raw)
    if errors:
        msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        raise ConfigurationError(msg)

    return Settings(raw)
