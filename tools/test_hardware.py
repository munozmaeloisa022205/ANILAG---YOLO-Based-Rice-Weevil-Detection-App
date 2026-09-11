"""Hardware verification for Anilag: camera wiring, clock and the
detection display/recording path.

Runs without any camera attached (OpenCV simply returns no frames), so it is
safe on a development machine as well as on the Raspberry Pi 5.

    python tools/test_hardware.py
"""
import os, sys, types, tempfile, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['EMAIL_ENABLED'] = 'false'

FAIL = []
def check(name, cond, extra=''):
    # Keep stdout pure ASCII: writing non-ASCII (e.g. the degree sign) to a redirected
    # stream on Windows/cp1252 can terminate the process mid-run.
    line = ('  PASS  ' if cond else '  FAIL  ') + name + (f'   {extra}' if extra else '')
    print(line.encode('ascii', 'backslashreplace').decode('ascii'), flush=True)
    if not cond: FAIL.append(name)

def trace(msg):
    print(f'  ...  {msg}', flush=True)

# ------------------------------------------------------------------- UI wiring
print('=== UI: clock, detection display, recording ===')
fake = types.ModuleType('ultralytics')
class _YOLO:
    def __init__(self, *a, **k): self.names = {0: 'Sitophilus Oryzae'}
    def to(self, *a, **k): return self
    def __call__(self, *a, **k): return []
fake.YOLO = _YOLO
sys.modules['ultralytics'] = fake

from src.backend.database import DatabaseManager
import src.gui.main_window as mw
from PyQt5.QtWidgets import QApplication
tmp = tempfile.mkdtemp()
db = DatabaseManager(os.path.join(tmp, 'anilag.db'))
mw.get_database.__globals__['_db_instance'] = db
app = QApplication(sys.argv)
w = mw.MainWindow()

w.update_clock()
from datetime import datetime
now = datetime.now()
check('header clock shows date', now.strftime('%B %d, %Y') in w.header_clock_label.text(),
      w.header_clock_label.text())
check('header clock shows time', now.strftime('%I:%M') in w.header_clock_label.text())

# detection recorded during a scan
w.current_scan_id = 'scan_temp'
db.create_scan('scan_temp', '2026-08-06 17:00:00', '', '')
w.scan_images_dir = os.path.join(tmp, 'imgs'); os.makedirs(w.scan_images_dir, exist_ok=True)
w.is_scanning = True
w.last_log_time = None
w.current_scan_folder = os.path.join(tmp, 'scanfolder')
os.makedirs(w.current_scan_folder, exist_ok=True)
w.update_detection(2, 0.8, 'Detection')
rows = db.get_detections_by_scan('scan_temp')
check('detection recorded to DB', rows and rows[0]['weevil_count'] == 2,
      str(rows[0]['weevil_count']) if rows else 'no rows')

meta = w.save_scan_metadata() or {}
check('scan metadata has model info', 'model' in meta)
check('metadata json written to scan folder',
      os.path.exists(os.path.join(w.current_scan_folder, 'scan_metadata.json')))

# Losing the scan folder must not cost us the database record
w.current_scan_folder = os.path.join(tmp, 'gone', 'missing')
check('metadata still saved when scan folder is unavailable',
      'model' in (w.save_scan_metadata() or {}))
w.is_scanning = False

shutil.rmtree(tmp, ignore_errors=True)
print('\n' + ('ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} FAILURES: {FAIL}'))
sys.exit(1 if FAIL else 0)
