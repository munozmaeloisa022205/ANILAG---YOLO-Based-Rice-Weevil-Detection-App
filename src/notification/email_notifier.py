import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
from typing import Optional, List
import os
from dotenv import load_dotenv


class EmailNotifier:
    def __init__(self, config_file: str = 'config.env'):
        load_dotenv(config_file)
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.sender_email = os.getenv('EMAIL_SENDER', '')
        self.sender_password = os.getenv('EMAIL_PASSWORD', '').replace(' ', '')
        self.recipients = [r.strip() for r in os.getenv('EMAIL_RECIPIENT', '').split(',') if r.strip()]
        self.max_attachment_mb = float(os.getenv('EMAIL_MAX_ATTACHMENT_MB', '24'))
        email_enabled_env = os.getenv('EMAIL_ENABLED', 'true').lower()
        self.email_enabled = email_enabled_env in ('true', '1', 'yes', 'on')
        # Activity/LED notifications are opt-in so the inbox only gets the scan reports.
        self.activity_alerts = os.getenv('EMAIL_ACTIVITY_ALERTS', 'false').lower() in ('true', '1', 'yes', 'on')
        self.enabled = self.email_enabled and bool(self.sender_email and self.sender_password and self.recipients)

    @property
    def recipient_email(self) -> str:
        return ', '.join(self.recipients)

    def initialize(self) -> bool:
        if not self.enabled:
            print("Email notification disabled. Check config.env for credentials.")
            return False
        print(f"Email notifier configured: {self.sender_email} -> {self.recipient_email}")
        return True

    def send_email(self, subject: str, body: str, is_html: bool = False,
                   attachments: Optional[List[str]] = None) -> bool:
        if not self.enabled:
            print("Email notification disabled")
            return False

        print(f"send_email: subject='{subject}', attachments={len(attachments or [])}")
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))

            for path in attachments or []:
                if not os.path.exists(path):
                    print(f"Attachment not found, skipping: {path}")
                    continue
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > self.max_attachment_mb:
                    print(f"Attachment too large ({size_mb:.1f} MB > {self.max_attachment_mb} MB), skipping: {path}")
                    continue
                print(f"Attaching: {os.path.basename(path)} ({size_mb:.2f} MB)")
                part = MIMEBase('application', 'octet-stream')
                with open(path, 'rb') as f:
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                                f'attachment; filename="{os.path.basename(path)}"')
                msg.attach(part)

            print(f"Connecting to SMTP {self.smtp_server}:{self.smtp_port}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=60) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                print(f"Logging in as {self.sender_email}...")
                server.login(self.sender_email, self.sender_password)
                print("Sending message...")
                server.send_message(msg, from_addr=self.sender_email, to_addrs=self.recipients)

            print(f"Email sent successfully: {subject}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"Email authentication failed: {e}. "
                  "Gmail requires a 16-character App Password in EMAIL_PASSWORD, not the account password.")
            return False
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
        if not self.activity_alerts:
            return True
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

    def send_scan_report(self, scan_id: str, zip_path: str, summary: dict) -> bool:
        """Email the end-of-scan report with the detection log and captured images zipped."""
        from datetime import datetime
        temp = summary.get('average_temperature_celsius')
        temp_str = f"{temp:.2f}°C" if isinstance(temp, (int, float)) else "N/A"
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024) if os.path.exists(zip_path) else 0

        # Required format: "ANILAG Detection Log - <date the mail is sent>"
        subject = f"ANILAG Detection Log - {datetime.now().strftime('%B %d, %Y')}"
        body = f"""
Anilag Rice Weevil Detection System
====================================
Detection Log Report

Scan ID: {scan_id}
Start Time: {summary.get('scan_start_time', 'N/A')}
End Time: {summary.get('scan_end_time', 'N/A')}
Duration: {summary.get('duration_seconds', 'N/A')} seconds

Results:
- Max Rice Weevil Count: {summary.get('max_weevil_count', 0)}
- Total Detection Log Entries: {summary.get('log_entry_count', 0)}
- Captured Detection Images: {summary.get('image_count', 0)}
- Average Temperature: {temp_str}
- Final Recommendation: {summary.get('recommendation', 'N/A')}

Attached: {os.path.basename(zip_path)} ({zip_size_mb:.2f} MB)
The archive contains the detection log (CSV), scan metadata (JSON), the captured
images of the rice weevils detected during this scan, and the recorded videos.

This is an automated notification from the Anilag detection system.
"""
        return self.send_email(subject, body, attachments=[zip_path])

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
