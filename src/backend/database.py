"""
Anilag Backend Database Module
Optimized SQLite database for Raspberry Pi 5
Stores detection data and scan metadata
"""

import sqlite3
import os
from datetime import datetime, timedelta
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
        """Create tables and indexes from the SQL schema file."""
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
    
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
    
    def add_scan_image(self, scan_id: str, timestamp: str, camera: str, filename: str,
                       image_blob: bytes, weevil_count: int = 0,
                       confidence_avg: Optional[float] = None,
                       file_path: Optional[str] = None,
                       detection_id: Optional[int] = None) -> int:
        """Store a captured image of detected rice weevils as a BLOB in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scan_images (scan_id, detection_id, timestamp, camera, weevil_count,
                                         confidence_avg, filename, file_path, image_bytes, image_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (scan_id, detection_id, timestamp, camera, weevil_count, confidence_avg,
                 filename, file_path, len(image_blob), sqlite3.Binary(image_blob))
            )
            return cursor.lastrowid

    def get_scan_images(self, scan_id: str, include_blob: bool = True) -> List[Dict[str, Any]]:
        """Get the captured images for a scan. Set include_blob=False to list metadata only."""
        columns = ("id, scan_id, detection_id, timestamp, camera, weevil_count, confidence_avg, "
                   "filename, file_path, image_bytes")
        if include_blob:
            columns += ", image_blob"
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT {columns} FROM scan_images WHERE scan_id=? ORDER BY timestamp ASC, camera ASC",
                (scan_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_scan_image_count(self, scan_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) AS n FROM scan_images WHERE scan_id=?", (scan_id,))
            return cursor.fetchone()['n']

    def create_report(self, scan_id: str, recipients: str, archive_name: str,
                      archive_bytes: int, image_count: int, log_entry_count: int,
                      included_videos: bool) -> int:
        """Record a scan report awaiting delivery."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scan_reports (scan_id, created_at, recipients, archive_name, archive_bytes,
                                          image_count, log_entry_count, included_videos, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (scan_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), recipients, archive_name,
                 archive_bytes, image_count, log_entry_count, 1 if included_videos else 0)
            )
            return cursor.lastrowid

    def update_report_status(self, report_id: int, status: str, error: Optional[str] = None):
        """Mark a report as sent or failed."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE scan_reports SET status=?, error=?, sent_at=? WHERE id=?",
                (status, error, datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == 'sent' else None,
                 report_id)
            )

    def get_reports(self, scan_id: Optional[str] = None, status: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if scan_id:
            clauses.append("scan_id=?")
            params.append(scan_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM scan_reports {where} ORDER BY created_at DESC LIMIT ?", params
            )
            return [dict(row) for row in cursor.fetchall()]

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
    
    def get_scan_overview(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Scans joined with their detection/image/report counts, for the Scan History UI."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.*,
                       (SELECT COUNT(*) FROM detections d WHERE d.scan_id = s.scan_id) AS detection_count,
                       (SELECT COUNT(*) FROM scan_images i WHERE i.scan_id = s.scan_id) AS image_count,
                       (SELECT COALESCE(SUM(i.image_bytes), 0) FROM scan_images i WHERE i.scan_id = s.scan_id) AS image_total_bytes,
                       (SELECT r.status FROM scan_reports r WHERE r.scan_id = s.scan_id
                         ORDER BY r.created_at DESC LIMIT 1) AS report_status
                FROM scans s
                ORDER BY s.start_time DESC
                LIMIT ?
                """,
                (limit,)
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
        """Delete a scan and its detections, images and reports"""
        with self._get_connection() as conn:
            # Delete children first (foreign keys)
            conn.execute("DELETE FROM scan_images WHERE scan_id=?", (scan_id,))
            conn.execute("DELETE FROM scan_reports WHERE scan_id=?", (scan_id,))
            conn.execute("DELETE FROM detections WHERE scan_id=?", (scan_id,))
            # Delete scan
            cursor = conn.execute("DELETE FROM scans WHERE scan_id=?", (scan_id,))
            return cursor.rowcount > 0
    
    def cleanup_old_scans(self, days: int = 30) -> int:
        """Delete scans older than specified days, including their image BLOBs"""
        with self._get_connection() as conn:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            stale = "SELECT scan_id FROM scans WHERE start_time < ?"
            conn.execute(f"DELETE FROM scan_images WHERE scan_id IN ({stale})", (cutoff_date,))
            conn.execute(f"DELETE FROM scan_reports WHERE scan_id IN ({stale})", (cutoff_date,))
            conn.execute(f"DELETE FROM detections WHERE scan_id IN ({stale})", (cutoff_date,))
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
