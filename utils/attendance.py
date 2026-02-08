"""
Attendance tracking system with CSV/Excel export.
"""

import os
import csv
from datetime import datetime, timedelta


class AttendanceManager:
    """Manages attendance check-in/out with cooldown and export."""

    def __init__(self, config, database):
        self.db = database
        att_cfg = config.section("attendance")
        self.enabled = att_cfg.get("enabled", True)
        self.cooldown_minutes = att_cfg.get("cooldown_minutes", 30)
        self.export_format = att_cfg.get("export_format", "csv")
        self.auto_export = att_cfg.get("auto_export", True)
        self.export_dir = config.get("paths", "attendance_dir", default="data/attendance")
        os.makedirs(self.export_dir, exist_ok=True)

        # Track last detection time per person (in-memory cooldown)
        self._last_seen = {}

    def mark_attendance(self, name, confidence=0.0):
        """
        Mark attendance for a person if cooldown has elapsed.
        Returns True if attendance was recorded.
        """
        if not self.enabled:
            return False

        now = datetime.now()
        last = self._last_seen.get(name)
        if last and (now - last) < timedelta(minutes=self.cooldown_minutes):
            return False

        self._last_seen[name] = now
        self.db.check_in(name)
        return True

    def mark_checkout(self, name):
        """Mark checkout for a person."""
        if not self.enabled:
            return
        self.db.check_out(name)

    def get_today_attendance(self):
        """Get today's attendance records."""
        return self.db.get_attendance()

    def export_attendance(self, date=None, format_override=None):
        """
        Export attendance to CSV or Excel.
        Returns the file path of the exported file.
        """
        fmt = format_override or self.export_format
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        records = self.db.get_attendance(date)
        if not records:
            print(f"No attendance records for {date}")
            return None

        if fmt == "xlsx":
            return self._export_xlsx(records, date)
        return self._export_csv(records, date)

    def _export_csv(self, records, date):
        filepath = os.path.join(self.export_dir, f"attendance_{date}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Check In", "Check Out", "Date", "Duration"])
            for r in records:
                duration = ""
                if r.get("check_in") and r.get("check_out"):
                    try:
                        t_in = datetime.fromisoformat(r["check_in"])
                        t_out = datetime.fromisoformat(r["check_out"])
                        duration = str(t_out - t_in)
                    except (ValueError, TypeError):
                        pass
                writer.writerow([
                    r["name"],
                    r.get("check_in", ""),
                    r.get("check_out", ""),
                    r.get("date", date),
                    duration,
                ])
        print(f"Attendance exported to {filepath}")
        return filepath

    def _export_xlsx(self, records, date):
        try:
            import openpyxl
        except ImportError:
            print("openpyxl not installed, falling back to CSV")
            return self._export_csv(records, date)

        filepath = os.path.join(self.export_dir, f"attendance_{date}.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"Attendance {date}"
        ws.append(["Name", "Check In", "Check Out", "Date", "Duration"])
        for r in records:
            duration = ""
            if r.get("check_in") and r.get("check_out"):
                try:
                    t_in = datetime.fromisoformat(r["check_in"])
                    t_out = datetime.fromisoformat(r["check_out"])
                    duration = str(t_out - t_in)
                except (ValueError, TypeError):
                    pass
            ws.append([
                r["name"],
                r.get("check_in", ""),
                r.get("check_out", ""),
                r.get("date", date),
                duration,
            ])
        wb.save(filepath)
        print(f"Attendance exported to {filepath}")
        return filepath

    def export_range(self, start_date, end_date):
        """Export attendance for a date range."""
        records = self.db.get_attendance_range(start_date, end_date)
        if not records:
            print(f"No attendance records for {start_date} to {end_date}")
            return None
        filepath = os.path.join(self.export_dir, f"attendance_{start_date}_to_{end_date}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Check In", "Check Out", "Date", "Duration"])
            for r in records:
                duration = ""
                if r.get("check_in") and r.get("check_out"):
                    try:
                        t_in = datetime.fromisoformat(r["check_in"])
                        t_out = datetime.fromisoformat(r["check_out"])
                        duration = str(t_out - t_in)
                    except (ValueError, TypeError):
                        pass
                writer.writerow([
                    r["name"],
                    r.get("check_in", ""),
                    r.get("check_out", ""),
                    r.get("date", ""),
                    duration,
                ])
        print(f"Attendance range exported to {filepath}")
        return filepath
