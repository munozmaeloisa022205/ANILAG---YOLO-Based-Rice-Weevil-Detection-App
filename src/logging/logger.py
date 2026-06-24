import csv
import os
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class DetectionLog:
    timestamp: str
    rice_weevil_count: int
    temperature_celsius: Optional[float]
    recommendation: str
    activity: str


class DetectionLogger:
    def __init__(self, log_file: str = 'logs/detection_log.csv'):
        self.log_file = log_file
        self.logs: List[DetectionLog] = []
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def initialize(self) -> bool:
        try:
            file_exists = os.path.exists(self.log_file)
            with open(self.log_file, 'a', newline='') as f:
                if not file_exists:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Rice Weevil Count', 'Temperature (°C)', 'Recommendation', 'Activity'])
            return True
        except Exception as e:
            print(f"Logger initialization error: {e}")
            return False

    def generate_recommendation(self, rice_weevil_count: int, is_after_mixing: bool = False) -> str:
        if is_after_mixing:
            if rice_weevil_count > 0:
                return "Heat Treatment Needed"
            else:
                return "Take Out Rice"
        else:
            if rice_weevil_count > 0:
                return "Mix and Sift Rice"
            else:
                return "No Action Needed"

    def log_detection(self, rice_weewolf_count: int, temperature: Optional[float], 
                     is_after_mixing: bool = False, activity: str = "Detection") -> DetectionLog:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recommendation = self.generate_recommendation(rice_weewolf_count, is_after_mixing)
        
        log_entry = DetectionLog(
            timestamp=timestamp,
            rice_weevil_count=rice_weewolf_count,
            temperature_celsius=temperature,
            recommendation=recommendation,
            activity=activity
        )
        
        self.logs.append(log_entry)
        self._write_to_csv(log_entry)
        
        return log_entry

    def _write_to_csv(self, log_entry: DetectionLog):
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log_entry.timestamp,
                    log_entry.rice_weevil_count,
                    log_entry.temperature_celsius,
                    log_entry.recommendation,
                    log_entry.activity
                ])
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def get_recent_logs(self, count: int = 10) -> List[DetectionLog]:
        return self.logs[-count:]

    def get_all_logs(self) -> List[DetectionLog]:
        return self.logs.copy()

    def clear_logs(self):
        self.logs.clear()
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            self.initialize()

    def get_log_summary(self) -> dict:
        if not self.logs:
            return {
                'total_detections': 0,
                'total_weevils_detected': 0,
                'avg_temperature': None,
                'recommendations': {}
            }
        
        total_weevils = sum(log.rice_weevil_count for log in self.logs)
        temps = [log.temperature_celsius for log in self.logs if log.temperature_celsius is not None]
        avg_temp = sum(temps) / len(temps) if temps else None
        
        recommendations = {}
        for log in self.logs:
            recommendations[log.recommendation] = recommendations.get(log.recommendation, 0) + 1
        
        return {
            'total_detections': len(self.logs),
            'total_weevils_detected': total_weevils,
            'avg_temperature': avg_temp,
            'recommendations': recommendations
        }
