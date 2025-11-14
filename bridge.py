"""Async helpers for the lightweight Nicole bridge bot used in AMLK tests.

This module provides a minimal interface that mirrors the historical helper
used by the AMLK automation.  The real production bridge code lives inside
``nicole_telegram.py`` and ``nicole_bridge.py``; however, the AMLK test-suite
imports a ``bridge`` module with a simple help command.  The goal of this file
is to offer a tiny compatibility layer so the legacy tests continue to run
while the modern bridge stack keeps evolving.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Callable

Handler = Callable[..., Any]

# Default command map mirrors the structure expected by the AMLK tests.  In
# production these values are overwritten during initialisation, so keeping
# them simple ensures the compatibility layer remains stable.
COMMAND_MAP: Dict[str, Tuple[Handler | None, str]] = {
    "/start": (None, "start Nicole resonance"),
    "/help": (None, "list available commands"),
}


def build_main_keyboard() -> str:
    """Return a placeholder keyboard description for compatibility."""

    return "[start] [help]"


async def help_command(update: Any, context: Any) -> None:
    """Send a help message through the Telegram update mock.

    The AMLK tests patch ``COMMAND_MAP`` and ``build_main_keyboard`` to inject
    their own behaviour.  The function therefore only needs to build the help
    text dynamically from the command map and forward it to
    ``update.message.reply_text``.
    """

    del context  # Unused but kept for signature compatibility.

    lines = ["Welcome! Available commands:"]
    for command, (_, description) in sorted(COMMAND_MAP.items()):
        lines.append(f"{command} - {description}")

    text = "\n".join(lines)
    await update.message.reply_text(text)
