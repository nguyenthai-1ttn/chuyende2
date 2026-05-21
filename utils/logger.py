"""
utils/logger.py — Colour-coded, module-prefixed logger for the subtitle agent.
"""

import logging
import sys


# ANSI colour codes
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    _FMT = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
    _DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset  = _COLOURS["RESET"]
        self._style._fmt = f"{colour}{self._FMT}{reset}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with colour output on stderr."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger          # already configured

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColourFormatter(datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
