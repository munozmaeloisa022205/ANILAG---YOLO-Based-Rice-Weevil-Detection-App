-- Anilag database schema
-- SQLite tables for scan metadata and detection events

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
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    weevil_count INTEGER DEFAULT 0,
    temperature_celsius REAL,
    recommendation TEXT,
    activity TEXT DEFAULT 'Detection',
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_scan_id ON detections(scan_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_scans_start_time ON scans(start_time);
CREATE INDEX IF NOT EXISTS idx_scans_scan_id ON scans(scan_id);
