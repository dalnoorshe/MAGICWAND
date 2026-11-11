"""MAGICWAND — a paper-trading bot that refuses to lie about itself.

One wish. Zero lies.

This package contains no live execution path and no withdrawal capability.
It cannot move real money. Read docs/SAFETY.md before doing anything else.
"""

__version__ = "0.1.0"

from .config import Config

__all__ = ["Config", "__version__"]
