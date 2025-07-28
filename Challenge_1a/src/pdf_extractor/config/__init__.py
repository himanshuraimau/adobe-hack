"""Configuration management for PDF structure extraction"""

from .config import Config

# Create global config instance
config = Config()

__all__ = ["Config", "config"]