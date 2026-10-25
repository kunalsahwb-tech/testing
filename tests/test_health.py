"""Tests for AEGIS-AD health check and logging initialization."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aegis_ad import __version__
from aegis_ad.__main__ import run_health_check
from aegis_ad.logging_setup import setup_logging
from aegis_ad.settings import load_settings


def test_logging_initialization(tmp_path: Path):
    """Verify that logging setup configures root logger without error."""
    settings = load_settings()
    logger = setup_logging(settings)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "aegis_ad"


def test_health_check_succeeds(capsys: pytest.CaptureFixture):
    """Verify that the health check command succeeds and outputs foundation status."""
    exit_code = run_health_check()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"AEGIS-AD Foundation v{__version__}" in captured.out
    assert "Status: Healthy" in captured.out
    assert "Configuration: Loaded successfully" in captured.out
    assert "Notice: CCTV ingestion, tracking, and vision pipelines are not active in Stage 0." in captured.out


def test_health_check_fails_on_broken_config(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Verify that health check reports failure and returns non-zero when config is invalid."""
    monkeypatch.setenv("AEGIS_APP__ENV", "broken_env")
    exit_code = run_health_check()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "[HEALTH CHECK FAILED]" in captured.err
