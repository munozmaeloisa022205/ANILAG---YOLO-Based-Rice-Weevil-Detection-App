import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import os
from dotenv import load_dotenv


class EmailNotifier:
    def __init__(self, config_file: str = 'config.env'):
        load_dotenv(config_file)
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.sender_email = os.getenv('EMAIL_SENDER', '')
        self.sender_password = os.getenv('EMAIL_PASSWORD', '')
        self.recipient_email = os.getenv('EMAIL_RECIPIENT', '')
        email_enabled_env = os.getenv('EMAIL_ENABLED', 'true').lower()
        self.email_enabled = email_enabled_env in ('true', '1', 'yes', 'on')
        self.enabled = self.email_enabled and bool(self.sender_email and self.sender_password and self.recipient_email)

    def initialize(self) -> bool:
        if not self.enabled:
            print("Email notification disabled. Check config.env for credentials.")
            return False
        print(f"Email notifier configured: {self.sender_email} -> {self.recipient_email}")
        return True

    def send_email(self, subject: str, body: str, is_html: bool = False) -> bool:
        if not self.enabled:
            print("Email notification disabled")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"Email sent: {subject}")
            return True
        except Exception as e:
            print(f"Email send error: {e}")
            return False

    def send_detection_alert(self, timestamp: str, rice_weevil_count: int, 
                            temperature: Optional[float], recommendation: str, 
                            activity: str) -> bool:
        subject = f"Anilag Detection Alert - {timestamp}"
        
        temp_str = f"{temperature:.2f}°C" if temperature is not None else "N/A"
        
        body = f"""
Anilag Rice Weevil Detection System
====================================

Detection Details:
- Timestamp: {timestamp}
- Activity: {activity}
- Rice Weevil Count: {rice_weevil_count}
- Temperature: {temp_str}
- Recommendation: {recommendation}

This is an automated notification from the Anilag detection system.
"""
        
        return self.send_email(subject, body)

    def send_activity_log(self, activity: str, details: str) -> bool:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"Anilag Activity Log - {timestamp}"
        body = f"""
Anilag System Activity
======================

Timestamp: {timestamp}
Activity: {activity}

Details:
{details}

This is an automated notification from the Anilag detection system.
"""
        return self.send_email(subject, body)

    def send_system_alert(self, alert_type: str, message: str) -> bool:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"Anilag System Alert - {alert_type}"
        body = f"""
Anilag System Alert
===================

Timestamp: {timestamp}
Alert Type: {alert_type}

Message:
{message}

This is an automated notification from the Anilag detection system.
"""
        return self.send_email(subject, body)
