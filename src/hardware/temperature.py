"""
DS18B20 1-Wire temperature sensor for the Anilag rice weevil detection system.

Raspberry Pi 5 setup
--------------------
Add to /boot/firmware/config.txt (Bookworm path; older images use /boot/config.txt):

    dtoverlay=w1-gpio,gpiopin=4

then reboot. Wiring: DQ -> GPIO4 (header pin 7), VDD -> 3.3V, GND -> GND, with a
4.7k pull-up resistor between DQ and 3.3V. Prefer normal (non-parasitic) power.

Sensors appear as /sys/bus/w1/devices/28-*/w1_slave.

Why this module polls in a background thread
--------------------------------------------
Reading w1_slave triggers a fresh conversion and BLOCKS for up to 750 ms at 12-bit
resolution. Calling it from the Qt GUI thread on every detection cycle would stall
the interface. So a daemon thread polls the sensor and read_temperature() returns
the most recent cached value immediately.

The 1-Wire bus is known to be less stable on the Pi 5 than on earlier boards
(raspberrypi/linux issue #6917: sensors intermittently disappear or return empty
data), so this module validates every read, retries, and re-detects the device if
it vanishes rather than giving up permanently.
"""

import os
import threading
import time
from typing import Optional

W1_DEVICES_DIR = '/sys/bus/w1/devices/'

# DS18B20 specified operating range.
TEMP_MIN_C = -55.0
TEMP_MAX_C = 125.0
# 85.000 C exactly is the DS18B20 power-on-reset scratchpad value, i.e. a read
# taken before the first conversion finished - not a real measurement.
POWER_ON_RESET_C = 85.0


class TemperatureSensor:
    def __init__(self, device_id: Optional[str] = None, poll_interval: Optional[float] = None):
        self.device_id = device_id
        self.sensor_path = None
        self.initialized = False
        self.simulated = False

        self.poll_interval = poll_interval if poll_interval is not None else float(
            os.getenv('TEMP_POLL_INTERVAL', '2.0'))
        # A cached reading older than this is considered stale and reported as unknown.
        self.stale_after = float(os.getenv('TEMP_STALE_AFTER', '15.0'))
        self.reject_power_on_reset = os.getenv(
            'TEMP_REJECT_85C', 'true').lower() in ('true', '1', 'yes', 'on')

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._temperature: Optional[float] = None
        self._last_good_time: Optional[float] = None
        self._recent = []           # small window for median smoothing
        # Increments once per accepted conversion. Callers polling faster than
        # poll_interval use this to tell a fresh reading from a repeated cached one.
        self._sample_id = 0
        self._read_count = 0
        self._error_count = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ setup

    def _discover(self) -> bool:
        """Locate the sensor's w1_slave file. Safe to call repeatedly."""
        # An explicitly configured id wins, but only if it actually exists.
        if self.device_id:
            candidate = os.path.join(W1_DEVICES_DIR, self.device_id, 'w1_slave')
            if os.path.exists(candidate):
                self.sensor_path = candidate
                return True

        if not os.path.isdir(W1_DEVICES_DIR):
            self.last_error = ("1-Wire not enabled: /sys/bus/w1/devices missing. Add "
                               "'dtoverlay=w1-gpio,gpiopin=4' to /boot/firmware/config.txt and reboot.")
            return False

        try:
            # DS18B20 family code is 28.
            for device in sorted(os.listdir(W1_DEVICES_DIR)):
                if device.startswith('28-'):
                    path = os.path.join(W1_DEVICES_DIR, device, 'w1_slave')
                    if os.path.exists(path):
                        if device != self.device_id:
                            print(f"DS18B20 detected: {device}")
                        self.device_id = device
                        self.sensor_path = path
                        return True
        except OSError as e:
            self.last_error = f"Could not list 1-Wire devices: {e}"
            return False

        self.last_error = ("No DS18B20 (28-*) found on the 1-Wire bus. Check the 4.7k pull-up "
                           "between DQ and 3.3V and the GPIO4 wiring.")
        return False

    def initialize(self) -> bool:
        """Find the sensor and start background polling.

        Returns True when a real sensor is present. When none is found the module
        stays usable (read_temperature returns None) so the GUI still runs.
        """
        found = self._discover()
        self.initialized = found
        self.simulated = not found

        if found:
            # Prime the cache synchronously so the first UI update has a value.
            first = self._read_sensor_blocking()
            if first is not None:
                self._store(first)
            print(f"Temperature sensor ready: DS18B20 {self.device_id} "
                  f"(polling every {self.poll_interval:g}s)")
        else:
            print(f"Temperature sensor not available. {self.last_error or ''}".strip())

        self._start_polling()
        return found

    def _start_polling(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name='ds18b20-poll', daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while not self._stop.is_set():
            if self.sensor_path is None or not os.path.exists(self.sensor_path):
                # Sensor missing or hot-unplugged - keep trying to find it again.
                if self._discover():
                    self.initialized = True
                    self.simulated = False
                else:
                    self.initialized = False
                    self._stop.wait(max(self.poll_interval, 2.0))
                    continue

            try:
                value = self._read_sensor_blocking()
            except Exception as e:
                # The poll thread must never die - that would silently freeze the
                # temperature readout for the rest of the session.
                self.last_error = f"Unexpected sensor error: {e}"
                self._error_count += 1
                value = None
            if value is not None:
                self._store(value)
            self._stop.wait(self.poll_interval)

    # ------------------------------------------------------------------ reads

    def _read_sensor_blocking(self, attempts: int = 3) -> Optional[float]:
        """Read and validate one temperature in Celsius. Blocks up to ~750ms per attempt."""
        if not self.sensor_path:
            return None
        for attempt in range(attempts):
            try:
                with open(self.sensor_path, 'r') as f:
                    lines = f.read().splitlines()
            except OSError as e:
                self.last_error = f"1-Wire read failed: {e}"
                self._error_count += 1
                time.sleep(0.2)
                continue

            self._read_count += 1

            # The kernel driver emits two lines; the first ends in YES/NO for the CRC.
            if len(lines) < 2 or 't=' not in lines[1]:
                self.last_error = "Incomplete w1_slave data"
                self._error_count += 1
                time.sleep(0.2)
                continue

            if not lines[0].rstrip().endswith('YES'):
                self.last_error = "CRC check failed (NO) - check wiring and pull-up resistor"
                self._error_count += 1
                time.sleep(0.2)
                continue

            try:
                raw = int(lines[1].split('t=')[-1].strip())
            except ValueError:
                self.last_error = "Could not parse temperature value"
                self._error_count += 1
                continue

            # The driver reports milli-degrees Celsius.
            celsius = raw / 1000.0

            if not (TEMP_MIN_C <= celsius <= TEMP_MAX_C):
                self.last_error = f"Reading {celsius:.3f}C outside DS18B20 range - discarded"
                self._error_count += 1
                continue

            if self.reject_power_on_reset and raw == int(POWER_ON_RESET_C * 1000):
                # Exactly 85.000C is the power-on default, not a measurement.
                self.last_error = "Discarded 85.000C power-on-reset value"
                self._error_count += 1
                time.sleep(0.2)
                continue

            self.last_error = None
            return celsius

        return None

    def _store(self, celsius: float):
        with self._lock:
            self._recent.append(celsius)
            if len(self._recent) > 5:
                self._recent.pop(0)
            self._temperature = celsius
            self._last_good_time = time.monotonic()
            self._sample_id += 1

    # ----------------------------------------------------------------- public

    def read_temperature(self) -> Optional[float]:
        """Latest temperature in Celsius, or None if unavailable/stale. Never blocks."""
        with self._lock:
            if self._temperature is None or self._last_good_time is None:
                return None
            # >= so that a stale_after of 0 means "never trust the cache"; coarse
            # monotonic clocks can otherwise report an age of exactly 0.
            if time.monotonic() - self._last_good_time >= self.stale_after:
                return None
            return self._temperature

    def read_sample(self) -> tuple:
        """(sample_id, celsius) for the latest reading. The id only changes when a new
        conversion has actually completed, so callers polling faster than the sensor
        can avoid recording the same measurement many times over."""
        with self._lock:
            if self._temperature is None or self._last_good_time is None:
                return (self._sample_id, None)
            if time.monotonic() - self._last_good_time >= self.stale_after:
                return (self._sample_id, None)
            return (self._sample_id, self._temperature)

    def read_temperature_smoothed(self) -> Optional[float]:
        """Median of the recent readings - rejects single-sample spikes."""
        with self._lock:
            if not self._recent or self._last_good_time is None:
                return None
            if time.monotonic() - self._last_good_time >= self.stale_after:
                return None
            ordered = sorted(self._recent)
            return ordered[len(ordered) // 2]

    def read_temperature_fahrenheit(self) -> Optional[float]:
        temp_c = self.read_temperature()
        return (temp_c * 9 / 5) + 32 if temp_c is not None else None

    def get_status(self) -> dict:
        with self._lock:
            age = (time.monotonic() - self._last_good_time) if self._last_good_time else None
            return {
                'device_id': self.device_id,
                'available': self.initialized,
                'temperature_celsius': self._temperature,
                'reading_age_seconds': round(age, 1) if age is not None else None,
                'stale': age is not None and age > self.stale_after,
                'samples': self._sample_id,
                'reads': self._read_count,
                'errors': self._error_count,
                'poll_interval': self.poll_interval,
                'last_error': self.last_error,
            }

    def cleanup(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
