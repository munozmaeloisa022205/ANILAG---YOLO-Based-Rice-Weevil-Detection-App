-- Anilag database schema
-- SQLite tables for scan metadata, detection events, captured images and email reports.
-- SQLite (WAL mode) is used deliberately: the Pi 5 runs a single-writer embedded
-- workload, so a MySQL/MongoDB daemon would only add RAM and IPC overhead.

CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT UNIQUE NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    max_weevil_count INTEGER DEFAULT 0,
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
    recommendation TEXT,
    activity TEXT DEFAULT 'Detection',
    left_count INTEGER DEFAULT 0,
    right_count INTEGER DEFAULT 0,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

-- Captured images of the detected rice weevils. The JPEG bytes live in the
-- database so the emailing system can build the report archive from the DB alone.
CREATE TABLE IF NOT EXISTS scan_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    detection_id INTEGER,
    timestamp TEXT NOT NULL,
    camera TEXT NOT NULL,
    weevil_count INTEGER DEFAULT 0,
    confidence_avg REAL,
    filename TEXT NOT NULL,
    file_path TEXT,
    image_bytes INTEGER DEFAULT 0,
    image_blob BLOB,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id),
    FOREIGN KEY (detection_id) REFERENCES detections(id)
);

-- One row per emailed scan report, so failed sends can be found and retried.
CREATE TABLE IF NOT EXISTS scan_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    recipients TEXT,
    archive_name TEXT,
    archive_bytes INTEGER DEFAULT 0,
    image_count INTEGER DEFAULT 0,
    log_entry_count INTEGER DEFAULT 0,
    included_videos INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    error TEXT,
    sent_at TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_scan_id ON detections(scan_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_scans_start_time ON scans(start_time);
CREATE INDEX IF NOT EXISTS idx_scans_scan_id ON scans(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_images_scan_id ON scan_images(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_reports_scan_id ON scan_reports(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_reports_status ON scan_reports(status);
