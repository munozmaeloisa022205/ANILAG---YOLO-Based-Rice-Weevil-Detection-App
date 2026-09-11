"""
Loading Screen Widget
Displays logo during 5-second startup with smooth animations
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt5.QtGui import QPixmap, QFont, QColor, QImage
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

    def _crop_to_content(self, pixmap: QPixmap) -> QPixmap:
        """Crop transparent padding around the logo so only the artwork shows.

        Scans the pixmap's alpha channel for the bounding box of visible
        pixels and returns a cropped copy. A threshold is used (alpha > 10)
        so that stray near-invisible noise pixels (alpha = 1) at the edges of
        the source image do not extend the crop and make the logo appear
        off-center. If the pixmap has no alpha or is fully transparent, the
        original is returned unchanged.
        """
        if pixmap.isNull():
            return pixmap
        image = pixmap.toImage()
        if image.isNull() or not image.hasAlphaChannel():
            return pixmap
        width = image.width()
        height = image.height()
        min_x, min_y = width, height
        max_x, max_y = -1, -1
        # Threshold ignores near-transparent noise pixels (alpha <= 10) that
        # would otherwise skew the bounding box and offset the logo.
        threshold = 10
        for y in range(height):
            for x in range(width):
                if image.pixelColor(x, y).alpha() > threshold:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        if max_x < 0 or max_y < 0:
            # No visible content found
            return pixmap
        crop_rect = QRect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        return pixmap.copy(crop_rect)

    def init_ui(self):
        self.setWindowTitle("Anilag - Loading")
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
        """)

        # Size to fit within the available screen geometry (excludes taskbar).
        # The Pi 5 touchscreen at 1x scaling is 800x400.
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(500, int(screen.width() * 0.85))
        h = min(380, int(screen.height() * 0.9))
        self.setFixedSize(w, h)

        # Center on screen
        self._center_window()

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(40, 30, 40, 30)
        
        # Top stretch centers the content block vertically (paired with the
        # bottom stretch). Do NOT use layout.setAlignment(Qt.AlignCenter) here -
        # it conflicts with the stretch items and pushes the logo off-center.
        layout.addStretch()
        
        # Logo container - no box/border, transparent background.
        # The source logo has transparent padding around the actual artwork,
        # so the pixmap is cropped to its content bounding box before scaling
        # (see _crop_to_content). This ensures only the logo itself is shown.
        # Container sized so the total content fits within the 550px window
        # with room for top/bottom stretches to vertically center the block.
        self.logo_container = QLabel()
        self.logo_container.setAlignment(Qt.AlignCenter)
        self.logo_container.setFixedSize(140, 140)
        self.logo_container.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                padding: 0px;
            }
        """)

        if self.logo_path and os.path.exists(self.logo_path):
            pixmap = QPixmap(self.logo_path)
            # Crop transparent padding so only the logo artwork remains
            pixmap = self._crop_to_content(pixmap)
            # Scale cropped logo to fill the container while keeping aspect ratio
            scaled = pixmap.scaled(
                140, 140,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.logo_container.setPixmap(scaled)
        else:
            # Fallback: styled text logo
            self.logo_container.setText("ANILAG")
            self.logo_container.setFont(QFont("Arial", 64, QFont.Bold))
            self.logo_container.setStyleSheet("""
                QLabel {
                    color: #2E7D32;
                    background: transparent;
                    border: none;
                }
            """)
        
        layout.addWidget(self.logo_container, alignment=Qt.AlignCenter)
        
        # Add spacing after logo
        layout.addSpacing(20)
        
        # Tagline - word wrap so it fits the narrower window on the Pi 5.
        tagline_label = QLabel("Rice Weevil Detection and Control System")
        tagline_label.setFont(QFont("Arial", 13, QFont.Bold))
        tagline_label.setStyleSheet("color: #666666;")
        tagline_label.setAlignment(Qt.AlignCenter)
        tagline_label.setWordWrap(True)
        tagline_label.setMaximumWidth(500)
        layout.addWidget(tagline_label)

        # Add spacing after tagline
        layout.addSpacing(5)

        # Subtitle/version
        version_label = QLabel("v1.0.0")
        version_label.setFont(QFont("Arial", 11))
        version_label.setStyleSheet("color: #999999;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Add spacing after version
        layout.addSpacing(15)

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
        """Center the window in the available screen area (excludes the taskbar).

        self.screen().geometry() returns the full physical screen including the
        area covered by the Pi's desktop taskbar, so a window centered on it can
        overlap the taskbar. availableGeometry() excludes the taskbar.
        """
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        size = self.geometry()
        self.move(
            screen.x() + (screen.width() - size.width()) // 2,
            screen.y() + (screen.height() - size.height()) // 2
        )
    
    def setup_animations(self):
        """Setup fade-in animation for loading screen"""
        self.opacity_effect = QGraphicsDropShadowEffect(self)
        # Optional: Add opacity animation if desired
        pass
    
    def setup_timer(self):
        """Setup loading timer with smooth updates"""
        self.loading_duration = 5.0  # seconds — full splash screen duration
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
            (0, 15, "Initializing hardware...", "Opening cameras..."),
            (15, 30, "Loading detection models...", "Loading neural network weights..."),
            (30, 50, "Configuring cameras...", "Calibrating camera module..."),
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
            QTimer.singleShot(500, self.loading_complete.emit)
    
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
