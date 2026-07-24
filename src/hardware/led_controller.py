import time
from typing import Optional


class LEDController:
    def __init__(self, gpio_pin: int = 18, led_count: int = 60, brightness: int = 255):
        self.gpio_pin = gpio_pin
        self.led_count = led_count
        self.brightness = brightness
        self.strip = None
        self.initialized = False

    def initialize(self) -> bool:
        try:
            # Import rpi_ws281x only on Raspberry Pi
            from rpi_ws281x import PixelStrip, ws

            LED_CHANNEL = 0
            # WS2813 uses the same protocol as WS2812; fall back gracefully
            # if a WS2813-specific constant is not present in this rpi_ws281x build.
            LED_STRIP = getattr(ws, 'WS2813_STRIP', ws.WS2812_STRIP)

            self.strip = PixelStrip(
                num=self.led_count,
                pin=self.gpio_pin,
                freq_hz=800000,
                dma=10,
                invert=False,
                brightness=self.brightness,
                strip_type=LED_STRIP,
                channel=LED_CHANNEL
            )

            self.strip.begin()
            self.initialized = True
            return True
        except ImportError:
            print("rpi_ws281x not available. Running in simulation mode.")
            self.initialized = True
            return True
        except Exception as e:
            print(f"LED controller initialization error: {e}")
            return False

    def set_color(self, r: int, g: int, b: int):
        if not self.initialized:
            return
        
        try:
            if self.strip:
                for i in range(self.led_count):
                    self.strip.setPixelColor(i, self.strip.Color(r, g, b))
                self.strip.show()
            else:
                print(f"Simulation: Setting LEDs to RGB({r}, {g}, {b})")
        except Exception as e:
            print(f"LED color set error: {e}")

    def set_red(self):
        self.set_color(255, 0, 0)

    def set_white(self):
        self.set_color(255, 255, 255)

    def off(self):
        self.set_color(0, 0, 0)

    def cleanup(self):
        if self.strip:
            self.off()
