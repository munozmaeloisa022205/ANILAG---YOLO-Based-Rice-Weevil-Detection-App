"""
Loading Screen Widget
Displays logo during 30-second startup
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
import os


class LoadingScreen(QWidget):
    """Loading screen with logo and progress bar"""
    
    loading_complete = pyqtSignal()
    
    def __init__(self, logo_path: str = None):
        super().__init__()
        self.logo_path = logo_path
        self.init_ui()
        self.setup_timer()
    
    def init_ui(self):
        self.setWindowTitle("Anilag - Loading")
        self.setFixedSize(600, 500)
        self.setStyleSheet("background-color: #f5f5f5;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)
        
        # Logo label
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        if self.logo_path and os.path.exists(self.logo_path):
            pixmap = QPixmap(self.logo_path)
            scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback text if logo not found
            self.logo_label.setText("ANILAG")
            self.logo_label.setFont(QFont("Arial", 48, QFont.Bold))
            self.logo_label.setStyleSheet("color: #2E7D32;")
        
        layout.addWidget(self.logo_label)
        
        # Title label
        title_label = QLabel("Rice Weevil Detection System")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Loading text
        self.loading_label = QLabel("Initializing...")
        self.loading_label.setFont(QFont("Arial", 12))
        self.loading_label.setStyleSheet("color: #666;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_label = QLabel("Loading components...")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #888;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def setup_timer(self):
        """Setup 30-second loading timer"""
        self.loading_duration = 30  # seconds
        self.elapsed_time = 0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # Update every 100ms
    
    def update_progress(self):
        """Update progress bar and status"""
        self.elapsed_time += 0.1
        progress = int((self.elapsed_time / self.loading_duration) * 100)
        self.progress_bar.setValue(progress)
        
        # Update status messages based on progress
        if progress < 20:
            self.status_label.setText("Initializing hardware...")
        elif progress < 40:
            self.status_label.setText("Loading detection models...")
        elif progress < 60:
            self.status_label.setText("Configuring sensors...")
        elif progress < 80:
            self.status_label.setText("Setting up database...")
        elif progress < 100:
            self.status_label.setText("Preparing interface...")
        else:
            self.status_label.setText("Ready!")
            self.timer.stop()
            # Small delay before emitting signal
            QTimer.singleShot(500, self.loading_complete.emit)
    
    def closeEvent(self, event):
        """Clean up timer when closing"""
        if hasattr(self, 'timer'):
            self.timer.stop()
        event.accept()
