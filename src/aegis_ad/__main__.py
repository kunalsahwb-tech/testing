"""Command-line entrypoint and basic health check for AEGIS-AD."""

from __future__ import annotations

import sys
from aegis_ad import __version__
from aegis_ad.logging_setup import setup_logging
from aegis_ad.settings import ConfigurationError, load_settings


def run_health_check() -> int:
    """Execute Stage 0 foundation health check.

    Verifies configuration loading and logging initialization.
    Does NOT assert or claim CCTV ingestion, detection, or tracking features.
    """
    try:
        settings = load_settings()
        logger = setup_logging(settings)
    except ConfigurationError as exc:
        print(f"[HEALTH CHECK FAILED] Configuration error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[HEALTH CHECK FAILED] Unexpected initialization error: {exc}", file=sys.stderr)
        return 1

    app_name = settings.app.get("name", "AEGIS-AD")
    app_env = settings.app.get("env", "unknown")

    logger.info("AEGIS-AD Foundation Health Check initialized.")

    print(f"AEGIS-AD Foundation v{__version__}")
    print("Status: Healthy")
    print(f"Application: {app_name}")
    print(f"Environment: {app_env}")
    print(f"Configuration: Loaded successfully")
    print(f"Logging: Initialized ({settings.logging.get('level', 'INFO')})")
    print("Notice: Web dashboard and AI investigation features have not been implemented yet.")
    return 0


def main() -> None:
    """Entry point for python -m aegis_ad."""
    sys.exit(run_health_check())


if __name__ == "__main__":
    main()
