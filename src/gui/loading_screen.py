"""
Loading Screen Widget
Displays logo during 30-second startup with smooth animations
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QPixmap, QFont, QColor
import os
import sys


class LoadingScreen(QWidget):
    """Loading screen with logo, progress bar, and status updates"""
    
    loading_complete = pyqtSignal()
    
    def __init__(self, logo_path: str = None):
        super().__init__()
        
        # Auto-detect logo if not provided
        self.logo_path = logo_path or self._find_logo()
        
        self.init_ui()
        self.setup_animations()
        self.setup_timer()
    
    def _find_logo(self) -> str:
        """Auto-find logo from common project locations"""
        # Check same directory as script
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        
        # Common logo filenames and locations to check
        candidates = [
            # Same directory as running script
            os.path.join(script_dir, "logo.png"),
            os.path.join(script_dir, "logo.jpg"),
            os.path.join(script_dir, "assets", "logo.png"),
            os.path.join(script_dir, "assets", "logo.jpg"),
            os.path.join(script_dir, "images", "logo.png"),
            os.path.join(script_dir, "images", "logo.jpg"),
            os.path.join(script_dir, "resources", "logo.png"),
            os.path.join(script_dir, "resources", "logo.jpg"),
            # Parent directory
            os.path.join(script_dir, "..", "assets", "logo.png"),
            os.path.join(script_dir, "..", "images", "logo.png"),
            # Absolute common paths
            "/home/pi/Anilag/assets/logo.png",
            "/home/pi/Anilag/images/logo.png",
        ]
        
        for path in candidates:
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                return normalized
        
        return None
    
    def init_ui(self):
        self.setWindowTitle("Anilag - Loading")
        self.setFixedSize(600, 500)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
        """)
        
        # Center on screen
        self._center_window()
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(25)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo container with shadow effect
        self.logo_container = QLabel()
        self.logo_container.setAlignment(Qt.AlignCenter)
        self.logo_container.setFixedSize(220, 220)
        
        if self.logo_path and os.path.exists(self.logo_path):
            pixmap = QPixmap(self.logo_path)
            # Scale to fit container while maintaining aspect ratio
            scaled = pixmap.scaled(
                200, 200, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.logo_container.setPixmap(scaled)
            
            # Add subtle drop shadow to logo
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 40))
            shadow.setOffset(0, 4)
            self.logo_container.setGraphicsEffect(shadow)
        else:
            # Fallback: styled text logo
            self.logo_container.setText("ANILAG")
            self.logo_container.setFont(QFont("Arial", 52, QFont.Bold))
            self.logo_container.setStyleSheet("""
                QLabel {
                    color: #2E7D32;
                    background-color: transparent;
                }
            """)
        
        layout.addWidget(self.logo_container, alignment=Qt.AlignCenter)
        
        # Title
        title_label = QLabel("Rice Weevil Detection System")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setStyleSheet("color: #333333;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle/version
        version_label = QLabel("v1.0.0")
        version_label.setFont(QFont("Arial", 11))
        version_label.setStyleSheet("color: #999999;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        # Loading status text
        self.loading_label = QLabel("Initializing...")
        self.loading_label.setFont(QFont("Arial", 13))
        self.loading_label.setStyleSheet("color: #666666;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                text-align: center;
                background-color: #ffffff;
                color: #333333;
                font-weight: bold;
                font-size: 12px;
                height: 28px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #66BB6A,
                    stop: 1 #4CAF50
                );
                border-radius: 8px;
                margin: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Detailed status
        self.status_label = QLabel("Loading components...")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #888888;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Spacer at bottom
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _center_window(self):
        """Center the window on the screen"""
        screen = self.screen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def setup_animations(self):
        """Setup fade-in animation for loading screen"""
        self.opacity_effect = QGraphicsDropShadowEffect(self)
        # Optional: Add opacity animation if desired
        pass
    
    def setup_timer(self):
        """Setup 30-second loading timer with smooth updates"""
        self.loading_duration = 30.0  # seconds
        self.elapsed_time = 0.0
        self.update_interval = 100  # ms
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(self.update_interval)
    
    def update_progress(self):
        """Update progress bar and status messages"""
        self.elapsed_time += self.update_interval / 1000.0
        progress = min(int((self.elapsed_time / self.loading_duration) * 100), 100)
        self.progress_bar.setValue(progress)
        
        # Status messages mapped to progress ranges
        status_map = [
            (0, 15, "Initializing hardware...", "Detecting sensors..."),
            (15, 30, "Loading detection models...", "Loading neural network weights..."),
            (30, 50, "Configuring sensors...", "Calibrating camera module..."),
            (50, 70, "Setting up database...", "Connecting to local storage..."),
            (70, 85, "Preparing interface...", "Loading UI components..."),
            (85, 98, "Finalizing setup...", "Performing system checks..."),
            (98, 100, "Ready!", "Launching application..."),
        ]
        
        for min_p, max_p, main_status, detail_status in status_map:
            if min_p <= progress < max_p or (progress == 100 and min_p == 98):
                self.loading_label.setText(main_status)
                self.status_label.setText(detail_status)
                break
        
        if progress >= 100:
            self.timer.stop()
            self.loading_label.setText("Ready!")
            self.status_label.setText("Launching application...")
            # Brief pause before emitting completion signal
            QTimer.singleShot(800, self.loading_complete.emit)
    
    def closeEvent(self, event):
        """Clean up timer when closing"""
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        event.accept()


# ============================================================================
# USAGE EXAMPLE - Add this to your main application file
# ============================================================================

"""
# main.py example integration:

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from loading_screen import LoadingScreen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anilag - Rice Weevil Detection")
        self.setGeometry(100, 100, 1024, 768)
        
        # Your main application UI here
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel("Main Application Running"))
        self.setCentralWidget(central)

def main():
    app = QApplication(sys.argv)
    
    # Create and show loading screen
    loading = LoadingScreen()  # Auto-detects logo, or pass path: LoadingScreen("path/to/logo.png")
    loading.show()
    
    # Create main window (but don't show yet)
    main_window = MainWindow()
    
    # Connect loading complete signal to show main window
    def on_loading_complete():
        loading.close()
        main_window.show()
        # Optional: Maximize for Raspberry Pi touchscreen
        # main_window.showMaximized()
    
    loading.loading_complete.connect(on_loading_complete)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
"""
