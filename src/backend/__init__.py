"""
Anilag Backend Module
Provides database storage for detection data and scan metadata
"""

from .database import DatabaseManager, get_database

__all__ = ['DatabaseManager', 'get_database']
