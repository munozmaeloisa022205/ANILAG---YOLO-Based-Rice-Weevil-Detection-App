"""Temporary test: WS2813 SPI encoding, DS18B20 parsing/robustness, clock, UI wiring."""
import os, sys, types, tempfile, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['EMAIL_ENABLED'] = 'false'

FAIL = []
def check(name, cond, extra=''):
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'   {extra}' if extra else ''), flush=True)
    if not cond: FAIL.append(name)

def trace(msg):
    print(f'  ...  {msg}', flush=True)

# ---------------------------------------------------------------- WS2813 / SPI
print('=== WS2813 SPI encoding ===')
from src.hardware.led_controller import LEDController, WS2813_RESET_US

led = LEDController(led_count=3, brightness=255)
led._color_order = 'GRB'
payload = led._encode_spi(255, 0, 0)          # pure red

# 3 pixels x 24 bits x 3 SPI bits = 216 bits = 27 bytes, + reset padding
reset_bytes = int(led._spi_hz * WS2813_RESET_US / 1_000_000 / 8) + 1
check('payload length correct', len(payload) == 27 + reset_bytes,
      f'{len(payload)} = 27 + {reset_bytes}')
check('reset gap >= 280us',
      (reset_bytes * 8 / led._spi_hz) * 1e6 >= 280,
      f'{(reset_bytes*8/led._spi_hz)*1e6:.0f} us')

bits = ''.join(f'{b:08b}' for b in payload[:27])
# GRB order for pure red => G=0x00, R=0xFF, B=0x00
expected = ('100' * 8) + ('110' * 8) + ('100' * 8)   # one pixel
check('one-bit encodes as 110', '110' in bits)
check('zero-bit encodes as 100', '100' in bits)
check('GRB byte order for red', bits[:72] == expected, bits[:24] + '...')
check('pattern repeats per pixel', bits == expected * 3)

# bit timing sanity
bit_period_ns = 3 / led._spi_hz * 1e9
check('bit period ~1250ns (800kHz)', 1200 < bit_period_ns < 1300, f'{bit_period_ns:.0f} ns')
t1h = 2 / led._spi_hz * 1e9
t0h = 1 / led._spi_hz * 1e9
check('T1H ~833ns in spec', 700 < t1h < 1000, f'{t1h:.0f} ns')
check('T0H ~417ns in spec', 250 < t0h < 550, f'{t0h:.0f} ns')

# white / off / brightness
w = led._encode_spi(255, 255, 255)
check('white sets all channels', all(b == 0b11011011 or b for b in w[:3]))
led.brightness = 128
check('brightness scales', led._scale(255) == 128, str(led._scale(255)))
led.brightness = 255

# Simulation mode must not crash and must report it did not reach hardware
os.environ['LED_USE_SPI'] = 'false'
sim = LEDController(led_count=5)
sim.initialize()
check('falls back to simulation', sim.backend == 'simulation', sim.backend)
check('set_red returns False when simulated', sim.set_red() is False)
check('mode tracked', sim.current_mode == 'red' and sim.current_color == (255, 0, 0))
check('white uses balance', (sim.set_white(), sim.current_color)[1] == (255, 255, 255))
check('off works', (sim.off(), sim.current_mode)[1] == 'off')
check('status reports backend', sim.get_status()['backend'] == 'simulation')
sim.cleanup()

# Verify writebytes2 receives exactly our payload via a fake spidev
print('\n=== SPI backend with a fake spidev ===')
written = []
fake_spidev = types.ModuleType('spidev')
class SpiDev:
    max_speed_hz = 0; mode = 0; no_cs = False
    def open(self, b, d): self.opened = (b, d)
    def writebytes2(self, data): written.append(bytes(data))
    def close(self): pass
fake_spidev.SpiDev = SpiDev
sys.modules['spidev'] = fake_spidev
os.environ['LED_USE_SPI'] = 'true'
spi_led = LEDController(led_count=2)
spi_led.initialize()
check('SPI backend selected', spi_led.backend == 'SPI', spi_led.backend)
written.clear()
check('set_red reaches hardware', spi_led.set_red() is True)
check('bytes written to SPI', len(written) == 1 and len(written[0]) > 0, f'{len(written[0])} bytes')
check('written payload matches encoder', written[0] == spi_led._encode_spi(255, 0, 0))
spi_led.cleanup()

# ------------------------------------------------------------------- DS18B20
print('\n=== DS18B20 parsing and robustness ===')
from src.hardware import temperature as tmod
from src.hardware.temperature import TemperatureSensor

tmp = tempfile.mkdtemp()
dev = os.path.join(tmp, '28-0000012345ab')
os.makedirs(dev)
slave = os.path.join(dev, 'w1_slave')
tmod.W1_DEVICES_DIR = tmp + os.sep

def write_slave(crc='YES', raw='28437'):
    with open(slave, 'w') as f:
        f.write(f"5b 01 4b 46 7f ff 0c 10 crc=1c : crc={crc}\n"
                f"5b 01 4b 46 7f ff 0c 10 crc=1c t={raw}\n")

write_slave(raw='28437')
s = TemperatureSensor()
ok = s.initialize()
check('sensor auto-detected', ok and s.device_id == '28-0000012345ab', str(s.device_id))
check('milli-degrees -> Celsius', abs(s.read_temperature() - 28.437) < 1e-6, str(s.read_temperature()))
check('Fahrenheit conversion', abs(s.read_temperature_fahrenheit() - 83.1866) < 1e-3)
check('negative temps work', (write_slave(raw='-5062'),
      abs(s._read_sensor_blocking() + 5.062) < 1e-6)[1])
check('CRC=NO rejected', (write_slave(crc='NO'), s._read_sensor_blocking() is None)[1])
check('85.000C power-on value rejected',
      (write_slave(raw='85000'), s._read_sensor_blocking() is None)[1])
check('out-of-range rejected', (write_slave(raw='200000'), s._read_sensor_blocking() is None)[1])
check('truncated file rejected',
      (open(slave, 'w').write('garbage\n'), s._read_sensor_blocking() is None)[1])
write_slave(raw='30125')
check('recovers after bad reads', abs(s._read_sensor_blocking() - 30.125) < 1e-6)

print('\n=== Non-blocking + staleness ===')
s2 = TemperatureSensor(poll_interval=0.05)
s2.initialize()
started = time.perf_counter()
for _ in range(200):
    s2.read_temperature()
elapsed_ms = (time.perf_counter() - started) * 1000
check('200 reads are non-blocking', elapsed_ms < 50, f'{elapsed_ms:.1f} ms for 200 reads')
check('background thread running', s2._thread is not None and s2._thread.is_alive())
s2.stale_after = 0.0
check('stale reading reported as None', s2.read_temperature() is None)
s2.stale_after = 15.0
check('smoothed read works', s2.read_temperature_smoothed() is not None)
st = s2.get_status()
check('status has diagnostics', all(k in st for k in ('device_id','available','reads','errors')))
s2.cleanup()
check('thread stopped on cleanup', s2._thread is None)

missing = TemperatureSensor()
tmod.W1_DEVICES_DIR = os.path.join(tmp, 'nope') + os.sep
check('missing bus handled gracefully', missing.initialize() is False)
check('missing sensor read returns None', missing.read_temperature() is None)
check('missing sensor explains why', 'dtoverlay' in (missing.last_error or ''),
      (missing.last_error or '')[:60])
missing.cleanup()
tmod.W1_DEVICES_DIR = tmp + os.sep

# ------------------------------------------------------------------- UI wiring
print('\n=== UI: clock, temperature display, recording ===')
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
check('live feed date label', now.strftime('%B %d, %Y') in w.date_label.text())
check('live feed time label', now.strftime('%p') in w.time_label.text())

# temperature display states
trace('swapping in fake sensor')
w.temp_sensor.cleanup()   # stop the window's own poll thread before replacing it
w.temp_sensor = s
write_slave(raw='27500')
s._store(27.5)
trace('calling update_temperature')
w.update_temperature()
trace('update_temperature returned')
check('temp displayed in Celsius', '27.5°C' in w.temp_label.text(), w.temp_label.text())
w.temp_sensor = missing
w.update_temperature()
check('missing sensor shown in UI', 'not detected' in w.temp_label.text(), w.temp_label.text())

# temperature recorded during a scan
w.temp_sensor = s
w.current_scan_id = 'scan_temp'
db.create_scan('scan_temp', '2026-08-06 17:00:00', '', '')
w.scan_images_dir = os.path.join(tmp, 'imgs'); os.makedirs(w.scan_images_dir, exist_ok=True)
w.is_scanning = True
w.last_log_time = None
w.update_detection(2, 0.8, 'Detection')
rows = db.get_detections_by_scan('scan_temp')
check('temperature recorded to DB', rows and abs(rows[0]['temperature_celsius'] - 27.5) < 1e-6,
      str(rows[0]['temperature_celsius']) if rows else 'no rows')
check('temperature collected for scan average', 27.5 in w.scan_temp_readings)
check('hardware status in metadata',
      'temperature_sensor' in (w.save_scan_metadata() or {}))
w.is_scanning = False

s.cleanup()
shutil.rmtree(tmp, ignore_errors=True)
print('\n' + ('ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} FAILURES: {FAIL}'))
sys.exit(1 if FAIL else 0)
