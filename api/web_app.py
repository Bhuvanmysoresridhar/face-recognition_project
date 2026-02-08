"""
Flask web dashboard for face recognition system.
Provides live feed, face management, detection history, and attendance views.
"""

import os
import json
import threading
from datetime import datetime

import cv2
from flask import (
    Flask, render_template, Response, request,
    redirect, url_for, flash, jsonify,
)

# Global reference set by create_app
_system = None
_camera_lock = threading.Lock()


def create_app(system, config):
    """
    Create and configure the Flask application.

    Args:
        system: The main FaceRecognitionSystem instance
        config: Config object
    """
    global _system
    _system = system

    web_cfg = config.section("web")
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
    )
    app.secret_key = web_cfg.get("secret_key", "change-this")

    # ---- Routes ----

    @app.route("/")
    def index():
        stats = {}
        if _system.db:
            stats["persons"] = len(_system.db.get_all_persons())
            stats["detections_today"] = len(
                _system.db.get_detections(
                    start_date=datetime.now().strftime("%Y-%m-%dT00:00:00")
                )
            )
            stats["attendance_today"] = len(_system.db.get_attendance())
        return render_template("index.html", stats=stats)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            _generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/faces")
    def faces():
        persons = []
        if _system.db:
            persons = _system.db.get_all_persons()
        known_dir = _system.engine.known_faces_dir
        for p in persons:
            # Find thumbnail
            person_dir = os.path.join(known_dir, p["name"])
            if os.path.isdir(person_dir):
                imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                p["images"] = len(imgs)
            else:
                img_file = None
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = os.path.join(known_dir, p["name"] + ext)
                    if os.path.exists(candidate):
                        img_file = candidate
                        break
                p["images"] = 1 if img_file else 0
        return render_template("faces.html", persons=persons)

    @app.route("/faces/add", methods=["POST"])
    def add_face():
        name = request.form.get("name", "").strip()
        if not name:
            flash("Name is required", "error")
            return redirect(url_for("faces"))

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Image is required", "error")
            return redirect(url_for("faces"))

        known_dir = _system.engine.known_faces_dir
        person_dir = os.path.join(known_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        existing = len([f for f in os.listdir(person_dir) if f.endswith((".jpg", ".png"))])
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        save_path = os.path.join(person_dir, f"{name}_{existing + 1}{ext}")
        file.save(save_path)

        # Reload faces
        _system.engine.load_known_faces()
        flash(f"Added image for {name}", "success")
        return redirect(url_for("faces"))

    @app.route("/faces/delete/<name>", methods=["POST"])
    def delete_face(name):
        import shutil
        known_dir = _system.engine.known_faces_dir
        person_dir = os.path.join(known_dir, name)
        if os.path.isdir(person_dir):
            shutil.rmtree(person_dir)
        else:
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(known_dir, name + ext)
                if os.path.exists(p):
                    os.remove(p)
        if _system.engine.cache:
            _system.engine.cache.remove_person(name)
        if _system.db:
            _system.db.remove_person(name)
        _system.engine.load_known_faces()
        flash(f"Removed {name}", "success")
        return redirect(url_for("faces"))

    @app.route("/detections")
    def detections():
        records = []
        stats = []
        if _system.db:
            records = _system.db.get_detections(limit=200)
            stats = _system.db.get_detection_stats()
        return render_template("detections.html", records=records, stats=stats)

    @app.route("/attendance")
    def attendance():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        records = []
        if _system.db:
            records = _system.db.get_attendance(date)
        return render_template("attendance.html", records=records, date=date)

    @app.route("/attendance/export")
    def export_attendance():
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        if _system.attendance:
            path = _system.attendance.export_attendance(date)
            if path:
                flash(f"Exported to {path}", "success")
            else:
                flash("No records to export", "warning")
        return redirect(url_for("attendance", date=date))

    @app.route("/api/stats")
    def api_stats():
        data = {"persons": 0, "detections": 0, "attendance": 0}
        if _system.db:
            data["persons"] = len(_system.db.get_all_persons())
            data["detections"] = len(
                _system.db.get_detections(
                    start_date=datetime.now().strftime("%Y-%m-%dT00:00:00")
                )
            )
            data["attendance"] = len(_system.db.get_attendance())
            data["detection_stats"] = _system.db.get_detection_stats()
        return jsonify(data)

    @app.route("/api/register", methods=["POST"])
    def api_register():
        """Register face from webcam via API."""
        name = request.json.get("name") if request.is_json else request.form.get("name")
        if not name:
            return jsonify({"error": "name required"}), 400
        # Capture frame from current camera
        if _system._video_capture and _system._video_capture.isOpened():
            with _camera_lock:
                ret, frame = _system._video_capture.read()
            if ret:
                success = _system.engine.register_face_from_frame(frame, name)
                if success:
                    return jsonify({"status": "registered", "name": name})
                return jsonify({"error": "No face detected in frame"}), 400
        return jsonify({"error": "Camera not available"}), 503

    return app


def _generate_frames():
    """Generate MJPEG frames from the recognition system."""
    while True:
        if _system and _system._latest_frame is not None:
            ret, buffer = cv2.imencode(".jpg", _system._latest_frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        else:
            # 1x1 black pixel placeholder
            blank = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blank + b"\r\n"
        import time
        time.sleep(0.033)  # ~30fps cap
