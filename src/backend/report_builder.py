"""
Anilag Scan Report Builder

Builds the end-of-scan report archive *from the database*. The detection log CSV and
the captured rice weevil images are read back out of SQLite (images are stored as
BLOBs), so the emailing system has a single source of truth and never depends on the
scan folder still being present on disk.
"""

import csv
import io
import json
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.backend.database import DatabaseManager

DETECTION_LOG_NAME = 'detection_log.csv'
METADATA_NAME = 'scan_metadata.json'
IMAGE_DIR_NAME = 'detected_images'
VIDEO_DIR_NAME = 'videos'


def build_detection_log_csv(detections: List[Dict], images_by_detection: Dict[Optional[int], List[str]]) -> str:
    """Render the detection log rows pulled from the database as CSV text."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['Timestamp', 'Rice Weevil Count', 'Temperature (C)',
                     'Recommendation', 'Activity', 'Captured Images'])
    for row in detections:
        temp = row.get('temperature_celsius')
        writer.writerow([
            row.get('timestamp', ''),
            row.get('weevil_count', 0),
            round(temp, 2) if isinstance(temp, (int, float)) else '',
            row.get('recommendation', ''),
            row.get('activity', ''),
            '; '.join(images_by_detection.get(row.get('id'), [])),
        ])
    return buffer.getvalue()


def build_scan_summary(db: DatabaseManager, scan_id: str) -> Optional[Dict]:
    """Assemble the scan summary used in the email body, entirely from the database."""
    scan = db.get_scan_by_id(scan_id)
    if not scan:
        return None

    detections = db.get_detections_by_scan(scan_id)
    images = db.get_scan_images(scan_id, include_blob=False)
    stats = db.get_detection_stats(scan_id)

    duration = None
    try:
        start = datetime.strptime(scan['start_time'], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(scan['end_time'], "%Y-%m-%d %H:%M:%S")
        duration = int((end - start).total_seconds())
    except (ValueError, TypeError, KeyError):
        pass

    summary = {
        'scan_id': scan_id,
        'scan_start_time': scan.get('start_time'),
        'scan_end_time': scan.get('end_time'),
        'duration_seconds': duration,
        'max_weevil_count': scan.get('max_weevil_count', 0),
        'average_temperature_celsius': scan.get('avg_temperature_celsius'),
        'temperature_readings_count': scan.get('temp_readings_count', 0),
        'log_entry_count': len(detections),
        'image_count': len(images),
        'image_total_bytes': sum(i.get('image_bytes') or 0 for i in images),
        'avg_weevil_count': round(stats['avg_count'], 2) if stats.get('avg_count') is not None else None,
        'recommendation': detections[-1].get('recommendation') if detections else 'No Action Needed',
    }

    if scan.get('metadata_json'):
        try:
            summary.update(json.loads(scan['metadata_json']))
            summary['scan_id'] = scan_id
        except json.JSONDecodeError:
            pass
    return summary


def build_scan_archive(db: DatabaseManager, scan_id: str, output_path: str,
                       summary: Optional[Dict] = None,
                       video_paths: Optional[List[str]] = None,
                       max_bytes: Optional[int] = None) -> Tuple[Optional[str], Dict]:
    """Write a zip containing the DB-stored detection log, metadata and weevil images.

    Videos are read from disk (they are far too large for BLOB storage) and are
    dropped if the archive would exceed max_bytes, so the log and images always fit.

    Returns (archive_path or None, info dict).
    """
    summary = summary or build_scan_summary(db, scan_id)
    if summary is None:
        return None, {'error': f'scan {scan_id} not found in database'}

    detections = db.get_detections_by_scan(scan_id)
    images = db.get_scan_images(scan_id, include_blob=True)

    images_by_detection: Dict[Optional[int], List[str]] = {}
    for image in images:
        images_by_detection.setdefault(image.get('detection_id'), []).append(image['filename'])

    csv_text = build_detection_log_csv(detections, images_by_detection)
    metadata_text = json.dumps(summary, indent=4, default=str)
    existing_videos = [p for p in (video_paths or []) if os.path.exists(p) and os.path.getsize(p) > 0]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def write_archive(include_videos: bool):
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(DETECTION_LOG_NAME, csv_text)
            zf.writestr(METADATA_NAME, metadata_text)
            for image in images:
                blob = image.get('image_blob')
                if blob:
                    zf.writestr(f"{IMAGE_DIR_NAME}/{image['filename']}", bytes(blob))
            if include_videos:
                for path in existing_videos:
                    zf.write(path, f"{VIDEO_DIR_NAME}/{os.path.basename(path)}")

    include_videos = bool(existing_videos)
    write_archive(include_videos)

    if include_videos and max_bytes and os.path.getsize(output_path) > max_bytes:
        include_videos = False
        write_archive(False)

    info = {
        'archive_path': output_path,
        'archive_name': os.path.basename(output_path),
        'archive_bytes': os.path.getsize(output_path),
        'image_count': len(images),
        'log_entry_count': len(detections),
        'included_videos': include_videos,
        'oversized': bool(max_bytes and os.path.getsize(output_path) > max_bytes),
    }
    return output_path, info
