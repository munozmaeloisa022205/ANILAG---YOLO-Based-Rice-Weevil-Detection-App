"""
WS2813 addressable LED controller for the Anilag rice weevil detection system.

Raspberry Pi 5 note
-------------------
The usual rpi_ws281x library does NOT work on the Pi 5. The Pi 5 routes its GPIOs
through the new RP1 south bridge, which is not compatible with the BCM PWM/DMA
peripheral that rpi_ws281x drives, so it fails at init with:

    ws2811_init failed with code -3 (Hardware revision is not supported)

This module therefore drives the strip over hardware SPI (MOSI = GPIO10, header
pin 19), which the RP1 exposes normally. Each WS2813 data bit is encoded as three
SPI bits at 2.4 MHz:

    bit 0 -> 100   (~417 ns high, ~833 ns low)
    bit 1 -> 110   (~833 ns high, ~417 ns low)

That yields a 1.25 us bit period (800 kHz), matching the WS2813 timing envelope.
rpi_ws281x is still used automatically on Pi 4 and older, and there is a
simulation mode so the GUI runs on a development machine.

Wiring (WS2813)
---------------
  DI  -> GPIO10 / header pin 19 (through a 3.3V->5V level shifter; the WS2813
         data threshold is ~0.7*VDD, so a bare 3.3V signal is marginal)
  BI  -> GND on the FIRST pixel only (backup data line; per the WS2813
         datasheet application circuit)
  5V  -> external 5V supply, NOT the Pi's 5V pin for more than a few pixels
  GND -> common ground shared with the Pi
Recommended: 330-470 ohm resistor in series with DI, and a 1000 uF capacitor
across 5V/GND at the strip.
"""

import os
import time
from typing import Optional, Tuple

# WS2813 needs a reset/latch gap of >280us (WS2812B only needs 50us).
WS2813_RESET_US = 300
SPI_BITS_PER_LED_BIT = 3
# 800 kHz data rate * 3 SPI bits per data bit
DEFAULT_SPI_HZ = 2_400_000


def _is_raspberry_pi_5() -> bool:
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi 5' in f.read()
    except OSError:
        return False


class LEDController:
    """Controls a WS2813 strip for the red (lure) and white (detection) lighting."""

    def __init__(self, gpio_pin: int = 18, led_count: int = 60, brightness: int = 255):
        self.gpio_pin = gpio_pin
        self.led_count = led_count
        self.brightness = max(0, min(255, brightness))
        self.strip = None          # rpi_ws281x PixelStrip, when used
        self._spi = None           # spidev.SpiDev, when used
        self.initialized = False
        self.backend = 'simulation'
        self.error: Optional[str] = None
        self.current_color: Tuple[int, int, int] = (0, 0, 0)
        self.current_mode = 'off'
        self._spi_bus = int(os.getenv('LED_SPI_BUS', '0'))
        self._spi_device = int(os.getenv('LED_SPI_DEVICE', '0'))
        self._spi_hz = int(os.getenv('LED_SPI_HZ', str(DEFAULT_SPI_HZ)))
        # WS2813 expects GRB byte order, like the WS2812/WS2812B.
        self._color_order = os.getenv('LED_COLOR_ORDER', 'GRB').upper()
        # White on an RGB strip is all three channels; allow trimming the tint.
        self._white_balance = tuple(
            int(v) for v in os.getenv('LED_WHITE_BALANCE', '255,255,255').split(',')[:3]
        )

    # ------------------------------------------------------------------ setup

    def initialize(self) -> bool:
        """Bring up the best available backend. Never raises; falls back to simulation."""
        prefer_spi = os.getenv('LED_USE_SPI', 'auto').lower()
        use_spi = prefer_spi in ('true', '1', 'yes', 'on') or (
            prefer_spi == 'auto' and _is_raspberry_pi_5())

        if use_spi and self._init_spi():
            return True
        if not use_spi and self._init_ws281x():
            return True
        # If the preferred backend failed, try the other one before simulating.
        if use_spi and self._init_ws281x():
            return True
        if not use_spi and self._init_spi():
            return True

        self.backend = 'simulation'
        self.initialized = True
        print(f"LED controller running in simulation mode ({self.error or 'no hardware backend'})")
        return True

    def _init_spi(self) -> bool:
        try:
            import spidev
        except ImportError:
            self.error = "spidev not installed (pip install spidev)"
            return False

        try:
            spi = spidev.SpiDev()
            spi.open(self._spi_bus, self._spi_device)
            spi.max_speed_hz = self._spi_hz
            spi.mode = 0
            spi.no_cs = True
            self._spi = spi
            self.backend = 'SPI'
            self.initialized = True
            print(f"LED controller initialized: WS2813 x{self.led_count} over SPI "
                  f"/dev/spidev{self._spi_bus}.{self._spi_device} @ {self._spi_hz} Hz "
                  f"(MOSI = GPIO10, header pin 19)")
            self.off()
            return True
        except Exception as e:
            self.error = (f"SPI open failed: {e}. Enable SPI with "
                          f"'sudo raspi-config' -> Interface Options -> SPI, then reboot.")
            print(f"LED controller: {self.error}")
            return False

    def _init_ws281x(self) -> bool:
        try:
            from rpi_ws281x import PixelStrip, ws
        except ImportError:
            self.error = "rpi_ws281x not available"
            return False

        try:
            # WS2813 shares the WS2812 protocol; use a WS2813 constant if this build has one.
            strip_type = getattr(ws, 'WS2813_STRIP', ws.WS2812_STRIP)
            strip = PixelStrip(
                num=self.led_count,
                pin=self.gpio_pin,
                freq_hz=800000,
                dma=10,
                invert=False,
                brightness=self.brightness,
                strip_type=strip_type,
                channel=0
            )
            strip.begin()
            self.strip = strip
            self.backend = 'rpi_ws281x'
            self.initialized = True
            print(f"LED controller initialized: WS2813 x{self.led_count} via rpi_ws281x "
                  f"on GPIO{self.gpio_pin}")
            self.off()
            return True
        except Exception as e:
            # On a Pi 5 this is the expected "Hardware revision is not supported" failure.
            self.error = f"rpi_ws281x init failed: {e}"
            print(f"LED controller: {self.error}")
            return False

    # ------------------------------------------------------------- SPI output

    def _order_channels(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        lookup = {'R': r, 'G': g, 'B': b}
        try:
            return tuple(lookup[c] for c in self._color_order)
        except KeyError:
            return (g, r, b)  # GRB default

    def _encode_spi(self, r: int, g: int, b: int) -> bytes:
        """Expand one colour, repeated across the strip, into an SPI bit pattern."""
        bits = []
        for byte in self._order_channels(r, g, b):
            for shift in range(7, -1, -1):
                # 0b110 for a 1 bit, 0b100 for a 0 bit
                bits.extend((1, 1, 0) if (byte >> shift) & 1 else (1, 0, 0))
        pixel_bits = bits * self.led_count

        # Pad to a whole number of bytes, then append the >280us reset gap.
        while len(pixel_bits) % 8:
            pixel_bits.append(0)
        payload = bytearray()
        for i in range(0, len(pixel_bits), 8):
            byte = 0
            for bit in pixel_bits[i:i + 8]:
                byte = (byte << 1) | bit
            payload.append(byte)

        reset_bytes = int(self._spi_hz * WS2813_RESET_US / 1_000_000 / 8) + 1
        payload.extend(b'\x00' * reset_bytes)
        return bytes(payload)

    def _scale(self, value: int) -> int:
        return max(0, min(255, value)) * self.brightness // 255

    # ---------------------------------------------------------------- control

    def set_color(self, r: int, g: int, b: int) -> bool:
        """Set every pixel to the same colour. Returns True if it reached hardware."""
        if not self.initialized:
            return False

        self.current_color = (r, g, b)
        try:
            if self._spi is not None:
                self._spi.writebytes2(self._encode_spi(self._scale(r), self._scale(g), self._scale(b)))
                # Hold the line low long enough for the strip to latch.
                time.sleep(WS2813_RESET_US / 1_000_000)
                return True
            if self.strip is not None:
                color = self.strip.Color(r, g, b)
                for i in range(self.led_count):
                    self.strip.setPixelColor(i, color)
                self.strip.show()
                return True
            print(f"Simulation: WS2813 x{self.led_count} -> RGB({r}, {g}, {b})")
            return False
        except Exception as e:
            print(f"LED color set error: {e}")
            return False

    def set_red(self) -> bool:
        """Red light - used to lure the rice weevils out."""
        self.current_mode = 'red'
        return self.set_color(255, 0, 0)

    def set_white(self) -> bool:
        """White light - used to illuminate the tray for detection."""
        self.current_mode = 'white'
        return self.set_color(*self._white_balance)

    def off(self) -> bool:
        self.current_mode = 'off'
        return self.set_color(0, 0, 0)

    def set_brightness(self, brightness: int):
        self.brightness = max(0, min(255, brightness))
        if self.strip is not None:
            self.strip.setBrightness(self.brightness)
        # Re-emit the current colour at the new brightness.
        self.set_color(*self.current_color)

    def get_status(self) -> dict:
        return {
            'backend': self.backend,
            'initialized': self.initialized,
            'led_count': self.led_count,
            'brightness': self.brightness,
            'mode': self.current_mode,
            'color': self.current_color,
            'color_order': self._color_order,
            'error': self.error,
        }

    def cleanup(self):
        try:
            if self.initialized:
                self.off()
        finally:
            if self._spi is not None:
                try:
                    self._spi.close()
                except Exception:
                    pass
                self._spi = None
            self.strip = None
            self.initialized = False
