"""
Notification system for email/SMS alerts on unknown faces or events.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NotificationManager:
    """Sends email alerts for security events (unknown faces, etc.)."""

    def __init__(self, config):
        notif_cfg = config.section("notifications")
        self.enabled = notif_cfg.get("enabled", False)
        self.unknown_face_alert = notif_cfg.get("unknown_face_alert", True)
        self.cooldown_minutes = notif_cfg.get("cooldown_minutes", 5)

        email_cfg = notif_cfg.get("email", {})
        self.smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = email_cfg.get("smtp_port", 587)
        self.sender = email_cfg.get("sender", "")
        self.password = email_cfg.get("password", "")
        self.recipients = email_cfg.get("recipients", [])

        self._last_alert = {}
        self._configured = bool(self.sender and self.password and self.recipients)

    def _can_send(self, event_key):
        """Check cooldown for an event type."""
        last = self._last_alert.get(event_key)
        if last and (datetime.now() - last) < timedelta(minutes=self.cooldown_minutes):
            return False
        return True

    def send_email(self, subject, body):
        """Send an email notification."""
        if not self._configured:
            logger.warning("Email not configured - skipping notification")
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "html"))

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def alert_unknown_face(self, camera_index=0, timestamp=None):
        """Send alert about an unknown face detection."""
        if not self.enabled or not self.unknown_face_alert:
            return False
        event_key = f"unknown_face_cam{camera_index}"
        if not self._can_send(event_key):
            return False

        ts = timestamp or datetime.now()
        subject = f"[Face Recognition] Unknown Face Detected - Camera {camera_index}"
        body = f"""
        <h2>Security Alert</h2>
        <p>An <strong>unknown face</strong> was detected.</p>
        <ul>
            <li><strong>Camera:</strong> {camera_index}</li>
            <li><strong>Time:</strong> {ts.strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
        <p>Please review the camera feed.</p>
        """
        sent = self.send_email(subject, body)
        if sent:
            self._last_alert[event_key] = datetime.now()
        return sent

    def alert_recognized_person(self, name, confidence, camera_index=0):
        """Send notification when a specific person is recognized."""
        if not self.enabled:
            return False
        event_key = f"recognized_{name}_cam{camera_index}"
        if not self._can_send(event_key):
            return False

        subject = f"[Face Recognition] {name} Detected - Camera {camera_index}"
        body = f"""
        <h2>Detection Notification</h2>
        <p><strong>{name}</strong> was recognized.</p>
        <ul>
            <li><strong>Confidence:</strong> {confidence:.1%}</li>
            <li><strong>Camera:</strong> {camera_index}</li>
            <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
        """
        sent = self.send_email(subject, body)
        if sent:
            self._last_alert[event_key] = datetime.now()
        return sent

    def send_daily_summary(self, attendance_records, detection_stats):
        """Send a daily summary email."""
        if not self.enabled:
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        subject = f"[Face Recognition] Daily Summary - {today}"

        people_rows = ""
        for rec in attendance_records:
            people_rows += f"<tr><td>{rec['name']}</td><td>{rec.get('check_in', '')}</td><td>{rec.get('check_out', 'N/A')}</td></tr>"

        stats_rows = ""
        for stat in detection_stats:
            stats_rows += f"<tr><td>{stat['name']}</td><td>{stat['count']}</td><td>{stat['avg_confidence']:.1%}</td></tr>"

        body = f"""
        <h2>Daily Summary - {today}</h2>
        <h3>Attendance</h3>
        <table border="1" cellpadding="5">
            <tr><th>Name</th><th>Check In</th><th>Check Out</th></tr>
            {people_rows or '<tr><td colspan="3">No records</td></tr>'}
        </table>
        <h3>Detection Statistics</h3>
        <table border="1" cellpadding="5">
            <tr><th>Name</th><th>Detections</th><th>Avg Confidence</th></tr>
            {stats_rows or '<tr><td colspan="3">No detections</td></tr>'}
        </table>
        """
        return self.send_email(subject, body)
