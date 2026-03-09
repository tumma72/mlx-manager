"""Modern launchctl API helpers.

Uses `launchctl bootstrap/bootout/kickstart/kill` instead of the deprecated
`launchctl load/unload/start/stop` commands (deprecated since macOS 10.11).
"""

import os
import subprocess

from loguru import logger


def get_gui_domain() -> str:
    """Get the GUI domain for the current user."""
    return f"gui/{os.getuid()}"


def get_service_target(label: str) -> str:
    """Get the full service target for a label."""
    return f"{get_gui_domain()}/{label}"


def bootstrap(plist_path: str, label: str) -> None:
    """Load a service via bootstrap (modern replacement for `launchctl load`).

    Idempotently bootouts the service first to avoid "already loaded" errors.
    """
    # Bootout first (idempotent — ignores "not found" errors)
    bootout(label)

    domain = get_gui_domain()
    result = subprocess.run(
        ["launchctl", "bootstrap", domain, plist_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"launchctl bootstrap failed: {result.stderr.strip()}")
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    logger.debug(f"Bootstrapped service {label}")


def bootout(label: str) -> None:
    """Unload a service via bootout (modern replacement for `launchctl unload`).

    Silently ignores "not found" errors for idempotent usage.
    """
    target = get_service_target(label)
    result = subprocess.run(
        ["launchctl", "bootout", target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Error 113 = "Could not find specified service" — expected when not loaded
        # Error 3 = "No such process" — also expected
        not_found = "113" in stderr or "3:" in stderr
        not_found = not_found or "No such process" in stderr or "Could not find" in stderr
        if not_found:
            logger.debug(f"Service {label} not loaded (bootout skipped)")
        else:
            logger.warning(f"launchctl bootout returned error: {stderr}")
    else:
        logger.debug(f"Booted out service {label}")


def kickstart(label: str) -> bool:
    """Start a service via kickstart (modern replacement for `launchctl start`)."""
    target = get_service_target(label)
    result = subprocess.run(
        ["launchctl", "kickstart", target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.debug(f"launchctl kickstart failed: {result.stderr.strip()}")
        return False
    return True


def kill_service(label: str, signal: int = 15) -> bool:
    """Stop a service via kill (modern replacement for `launchctl stop`).

    Args:
        label: The service label.
        signal: Signal number to send (default: 15 = SIGTERM).
    """
    target = get_service_target(label)
    result = subprocess.run(
        ["launchctl", "kill", str(signal), target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.debug(f"launchctl kill failed: {result.stderr.strip()}")
        return False
    return True
