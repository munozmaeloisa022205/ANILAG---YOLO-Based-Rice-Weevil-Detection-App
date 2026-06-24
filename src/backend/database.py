"""
Anilag Backend Database Module
Optimized SQLite database for Raspberry Pi 5
Stores detection data and scan metadata
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading


class DatabaseManager:
    """Thread-safe SQLite database manager optimized for Raspberry Pi 5"""
    
    def __init__(self, db_path: str = 'data/anilag.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_db_directory()
        self._initialize_database()
    
    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with thread safety"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Balanced safety/performance
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache for Pi 5
            conn.execute("PRAGMA temp_store=MEMORY")  # Use RAM for temp tables
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    def _initialize_database(self):
        """Create tables if they don't exist"""
        with self._get_connection() as conn:
            # Scans table - stores scan metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT UNIQUE NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    max_weevil_count INTEGER DEFAULT 0,
                    avg_temperature_celsius REAL,
                    temp_readings_count INTEGER DEFAULT 0,
                    left_video_path TEXT,
                    right_video_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Detections table - stores individual detection events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    weevil_count INTEGER DEFAULT 0,
                    temperature_celsius REAL,
                    recommendation TEXT,
                    activity TEXT DEFAULT 'Detection',
                    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_scan_id ON detections(scan_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_start_time ON scans(start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_scan_id ON scans(scan_id)")
    
    def create_scan(self, scan_id: str, start_time: str, left_video_path: str, 
                    right_video_path: str) -> int:
        """Create a new scan record"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scans (scan_id, start_time, end_time, left_video_path, right_video_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (scan_id, start_time, start_time, left_video_path, right_video_path)
            )
            return cursor.lastrowid
    
    def update_scan(self, scan_id: str, end_time: str, max_count: int, 
                    avg_temp: float, temp_readings_count: int, metadata_json: str = None):
        """Update scan record with final data"""
        with self._get_connection() as conn:
            if metadata_json:
                conn.execute(
                    """
                    UPDATE scans 
                    SET end_time=?, max_weevil_count=?, avg_temperature_celsius=?, 
                        temp_readings_count=?, metadata_json=?
                    WHERE scan_id=?
                    """,
                    (end_time, max_count, avg_temp, temp_readings_count, metadata_json, scan_id)
                )
            else:
                conn.execute(
                    """
                    UPDATE scans 
                    SET end_time=?, max_weevil_count=?, avg_temperature_celsius=?, 
                        temp_readings_count=?
                    WHERE scan_id=?
                    """,
                    (end_time, max_count, avg_temp, temp_readings_count, scan_id)
                )
    
    def add_detection(self, scan_id: str, timestamp: str, weevil_count: int, 
                      temperature: Optional[float], recommendation: str, 
                      activity: str = "Detection") -> int:
        """Add a detection record"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detections (scan_id, timestamp, weevil_count, temperature_celsius, 
                                       recommendation, activity)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scan_id, timestamp, weevil_count, temperature, recommendation, activity)
            )
            return cursor.lastrowid
    
    def get_scan_by_id(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan metadata by scan ID"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM scans WHERE scan_id=?",
                (scan_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_scans(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all scans with pagination"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM scans 
                ORDER BY start_time DESC 
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_detections_by_scan(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get all detections for a specific scan"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM detections 
                WHERE scan_id=? 
                ORDER BY timestamp ASC
                """,
                (scan_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent detections across all scans"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_detection_stats(self, scan_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for detections"""
        with self._get_connection() as conn:
            if scan_id:
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total_detections,
                        AVG(weevil_count) as avg_count,
                        MAX(weevil_count) as max_count,
                        AVG(temperature_celsius) as avg_temp
                    FROM detections 
                    WHERE scan_id=?
                    """,
                    (scan_id,)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total_detections,
                        AVG(weevil_count) as avg_count,
                        MAX(weevil_count) as max_count,
                        AVG(temperature_celsius) as avg_temp
                    FROM detections
                    """
                )
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def delete_scan(self, scan_id: str) -> bool:
        """Delete a scan and its detections"""
        with self._get_connection() as conn:
            # Delete detections first (foreign key)
            conn.execute("DELETE FROM detections WHERE scan_id=?", (scan_id,))
            # Delete scan
            cursor = conn.execute("DELETE FROM scans WHERE scan_id=?", (scan_id,))
            return cursor.rowcount > 0
    
    def cleanup_old_scans(self, days: int = 30) -> int:
        """Delete scans older than specified days"""
        with self._get_connection() as conn:
            cutoff_date = datetime.now().replace(day=datetime.now().day - days).strftime("%Y-%m-%d")
            cursor = conn.execute(
                "DELETE FROM scans WHERE start_time < ?", (cutoff_date,)
            )
            return cursor.rowcount
    
    def get_database_size(self) -> int:
        """Get database file size in bytes"""
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0
    
    def vacuum(self):
        """Optimize database by rebuilding it"""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
    
    def close(self):
        """Close database connections"""
        pass  # Connections are managed by context manager


# Singleton instance for application-wide use
_db_instance: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


def get_database(db_path: str = 'data/anilag.db') -> DatabaseManager:
    """Get singleton database instance"""
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            _db_instance = DatabaseManager(db_path)
        return _db_instance
