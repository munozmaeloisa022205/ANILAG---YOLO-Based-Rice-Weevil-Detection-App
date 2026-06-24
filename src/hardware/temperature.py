import time
from typing import Optional
import os


class TemperatureSensor:
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.sensor_path = None
        self.initialized = False

    def initialize(self) -> bool:
        try:
            base_dir = '/sys/bus/w1/devices/'
            if self.device_id:
                self.sensor_path = os.path.join(base_dir, self.device_id, 'w1_slave')
                if os.path.exists(self.sensor_path):
                    self.initialized = True
                    return True
                else:
                    print(f"Sensor device {self.device_id} not found")
                    return False
            else:
                # Auto-detect first available sensor
                if not os.path.exists(base_dir):
                    print("1-Wire devices directory not found. Enable 1-Wire interface in raspi-config")
                    return False
                
                for device in os.listdir(base_dir):
                    if device.startswith('28-'):
                        self.device_id = device
                        self.sensor_path = os.path.join(base_dir, device, 'w1_slave')
                        self.initialized = True
                        print(f"Auto-detected temperature sensor: {device}")
                        return True
                
                print("No DS18B20 sensor found")
                return False
        except Exception as e:
            print(f"Temperature sensor initialization error: {e}")
            return False

    def read_temperature(self) -> Optional[float]:
        if not self.initialized or not self.sensor_path:
            return None
        
        try:
            with open(self.sensor_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            # Check if CRC is valid (YES at end of first line)
            if 'YES' not in lines[0]:
                return None
            
            # Extract temperature from second line
            temp_str = lines[1].split('t=')[-1].strip()
            temp_celsius = float(temp_str) / 1000.0
            return temp_celsius
        except Exception as e:
            print(f"Temperature read error: {e}")
            return None

    def read_temperature_fahrenheit(self) -> Optional[float]:
        temp_c = self.read_temperature()
        if temp_c is not None:
            return (temp_c * 9/5) + 32
        return None
