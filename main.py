"""
Enhanced Face Recognition System
---------------------------------
Production-grade real-time face recognition with:
- Multi-image per person support
- Persistent encoding cache (no re-encoding on restart)
- Anti-spoofing / liveness detection
- Face tracking across frames
- Attendance system with CSV/Excel export
- SQLite database for detection history
- Email/SMS alert notifications
- Flask web dashboard
- CLI with full configuration control
- Optional FAISS indexing for large face databases
"""

import cv2
import threading
from datetime import datetime

from utils.config import Config
from utils.database import Database
from utils.encoding_cache import EncodingCache
from utils.attendance import AttendanceManager
from utils.notifications import NotificationManager
from recognition.engine import FaceRecognitionEngine
from recognition.liveness import LivenessDetector
from recognition.tracker import FaceTracker


class FaceRecognitionSystem:
    """Main orchestrator that wires all modules together."""

    def __init__(self, config_path="config.yaml"):
        self.config = Config(config_path)

        # Core modules
        self.db = Database(self.config.get("paths", "database"))
        cache = EncodingCache(self.config.get("paths", "encoding_cache"))
        self.engine = FaceRecognitionEngine(self.config, encoding_cache=cache, database=self.db)
        self.liveness = LivenessDetector(self.config)
        self.tracker = FaceTracker(self.config)
        self.attendance = AttendanceManager(self.config, self.db)
        self.notifications = NotificationManager(self.config)

        # Runtime state
        self._video_capture = None
        self._latest_frame = None
        self._running = False

    def run(self, camera_index=None):
        """Main recognition loop with all production features."""
        if camera_index is None:
            camera_index = self.config.get("camera", "index", default=0)

        cap = self._open_camera(camera_index)
        if cap is None:
            return
        self._video_capture = cap

        rec_cfg = self.config.section("recognition")
        frame_scale = rec_cfg.get("frame_scale", 0.25)
        skip_frames = rec_cfg.get("skip_frames", 2)
        inv_scale = int(1 / frame_scale)

        print(f"\n{'=' * 50}")
        print("  Face Recognition System Started")
        print(f"{'=' * 50}")
        print(f"  Threshold     : {self.engine.threshold}")
        print(f"  Known people  : {len(self.engine.person_encodings)}")
        print(f"  Encodings     : {len(self.engine.known_encodings)}")
        print(f"  Liveness      : {'ON' if self.liveness.enabled else 'OFF'}")
        print(f"  Tracking      : {'ON' if self.tracker.enabled else 'OFF'}")
        print(f"  Attendance    : {'ON' if self.attendance.enabled else 'OFF'}")
        print(f"{'=' * 50}")
        print("  Controls: q=quit  s=save  r=register  a=attendance")
        print(f"{'=' * 50}\n")

        self._running = True
        frame_count = 0
        process_counter = 0
        tracked_objects = {}

        import face_recognition as fr

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            process_counter += 1
            do_process = process_counter >= skip_frames

            if do_process:
                process_counter = 0
                small = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                raw_locations = fr.face_locations(rgb_small, model=self.engine.model)

                valid = []
                for (t, r, b, l) in raw_locations:
                    if (r - l) * inv_scale >= self.engine.min_face_size and \
                       (b - t) * inv_scale >= self.engine.min_face_size:
                        valid.append((t, r, b, l))

                encodings = fr.face_encodings(rgb_small, valid)

                names, confs = [], []
                for enc in encodings:
                    name, conf, dist = self.engine.recognize_face(enc)
                    names.append(name)
                    confs.append(conf)

                # Scale locations to full frame
                face_locations = [(t * inv_scale, r * inv_scale, b * inv_scale, l * inv_scale)
                                  for (t, r, b, l) in valid]

                tracked_objects = self.tracker.update(face_locations, names, confs)

                # Log and attend
                for name, conf in zip(names, confs):
                    if name != "Unknown":
                        self.db.log_detection(name, conf, camera_index=camera_index)
                        self.attendance.mark_attendance(name, conf)
                    elif self.notifications.enabled:
                        self.notifications.alert_unknown_face(camera_index)

                # Liveness checks
                if self.liveness.enabled:
                    for obj_id, (bbox, name, conf) in list(tracked_objects.items()):
                        result = self.liveness.check_liveness(frame, bbox, face_id=str(obj_id))
                        if not result["is_live"] and name != "Unknown":
                            tracked_objects[obj_id] = (bbox, f"{name}[SPOOF?]", conf)

            # Draw results
            for obj_id, (bbox, name, conf) in tracked_objects.items():
                top, right, bottom, left = bbox
                is_spoof = "[SPOOF?]" in name
                is_unknown = name == "Unknown"

                if is_spoof:
                    color = (0, 165, 255)  # orange
                elif is_unknown:
                    color = (0, 0, 255)    # red
                else:
                    color = (0, 255, 0)    # green

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = "Unknown" if is_unknown else f"{name} ({conf:.0%})"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(frame, (left, bottom - th - 12),
                              (left + tw + 12, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"ID:{obj_id}", (left, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Overlay
            info = f"Faces: {len(tracked_objects)} | Frame: {frame_count}"
            if self.liveness.enabled:
                info += " | Liveness: ON"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self._latest_frame = frame.copy()
            cv2.imshow("Face Recognition System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                print("Session data saved to database.")
            elif key == ord("r"):
                print("\n[REGISTER] Enter name: ", end="", flush=True)
                register_name = input().strip()
                if register_name:
                    success = self.engine.register_face_from_frame(frame, register_name)
                    print(f"[REGISTER] {'Success!' if success else 'No face detected.'}")
            elif key == ord("a"):
                self._print_attendance()

        # Cleanup
        self._running = False
        cap.release()
        self._video_capture = None
        cv2.destroyAllWindows()

        if self.attendance.auto_export:
            self.attendance.export_attendance()
        self._print_session_stats(frame_count)

    def run_web(self, host=None, port=None):
        """Start the Flask web dashboard."""
        from api.web_app import create_app
        web_cfg = self.config.section("web")
        host = host or web_cfg.get("host", "0.0.0.0")
        port = port or web_cfg.get("port", 5000)

        app = create_app(self, self.config)

        cam_thread = threading.Thread(target=self._background_recognition, daemon=True)
        cam_thread.start()

        print(f"\nWeb dashboard: http://{host}:{port}")
        app.run(host=host, port=port, debug=False, threaded=True)

    def _background_recognition(self):
        """Run recognition loop in background for web mode."""
        import face_recognition as fr

        camera_index = self.config.get("camera", "index", default=0)
        cap = self._open_camera(camera_index)
        if cap is None:
            return
        self._video_capture = cap

        rec_cfg = self.config.section("recognition")
        frame_scale = rec_cfg.get("frame_scale", 0.25)
        skip_frames = rec_cfg.get("skip_frames", 2)
        inv_scale = int(1 / frame_scale)

        self._running = True
        process_counter = 0
        tracked_objects = {}

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            process_counter += 1
            if process_counter >= skip_frames:
                process_counter = 0
                small = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                raw_locs = fr.face_locations(rgb_small, model=self.engine.model)
                valid = [(t, r, b, l) for (t, r, b, l) in raw_locs
                         if (r - l) * inv_scale >= self.engine.min_face_size
                         and (b - t) * inv_scale >= self.engine.min_face_size]

                encs = fr.face_encodings(rgb_small, valid)
                names, confs = [], []
                for enc in encs:
                    n, c, d = self.engine.recognize_face(enc)
                    names.append(n)
                    confs.append(c)

                full_locs = [(t * inv_scale, r * inv_scale, b * inv_scale, l * inv_scale)
                             for (t, r, b, l) in valid]
                tracked_objects = self.tracker.update(full_locs, names, confs)

                for n, c in zip(names, confs):
                    if n != "Unknown":
                        self.db.log_detection(n, c, camera_index=camera_index)
                        self.attendance.mark_attendance(n, c)

            # Draw
            for obj_id, (bbox, name, conf) in tracked_objects.items():
                top, right, bottom, left = bbox
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({conf:.0%})" if name != "Unknown" else "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            self._latest_frame = frame.copy()

        cap.release()
        self._video_capture = None

    def _open_camera(self, camera_index):
        """Try opening camera with multiple backends."""
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        w = self.config.get("camera", "width", default=640)
                        h = self.config.get("camera", "height", default=480)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                        print(f"Camera {camera_index} opened successfully")
                        return cap
                    cap.release()
                else:
                    cap.release()
            except Exception:
                continue
        print("\nERROR: Could not open webcam!")
        print("Run 'python webcam_test.py' to diagnose.")
        return None

    def _print_attendance(self):
        records = self.attendance.get_today_attendance()
        print(f"\n{'=' * 40}")
        print(f"  Attendance - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'=' * 40}")
        if records:
            for r in records:
                co = r.get("check_out") or "Present"
                if co != "Present":
                    co = co[:19]
                print(f"  {r['name']:20s} In: {r['check_in'][:19]}  Out: {co}")
        else:
            print("  No attendance records today.")
        print(f"{'=' * 40}\n")

    def _print_session_stats(self, frame_count):
        print(f"\n{'=' * 50}")
        print("  Session Statistics")
        print(f"{'=' * 50}")
        print(f"  Frames processed : {frame_count}")
        stats = self.db.get_detection_stats()
        total = sum(s["count"] for s in stats) if stats else 0
        print(f"  Total detections : {total}")
        if stats:
            print(f"  People detected  : {len(stats)}")
            for s in stats:
                print(f"    {s['name']:20s} x{s['count']}  (avg conf: {s['avg_confidence']:.0%})")
        attendance = self.attendance.get_today_attendance()
        if attendance:
            print(f"  Attendance today : {len(attendance)}")
        print(f"{'=' * 50}\n")

    def stop(self):
        self._running = False


if __name__ == "__main__":
    from cli import build_parser
    args = build_parser().parse_args()

    system = FaceRecognitionSystem(config_path=args.config)

    # Override config with CLI args
    if args.threshold is not None:
        system.engine.threshold = args.threshold
    if args.min_face_size is not None:
        system.engine.min_face_size = args.min_face_size
    if args.model:
        system.engine.model = args.model
    if args.no_liveness:
        system.liveness.enabled = False
    if args.no_tracking:
        system.tracker.enabled = False
    if args.no_attendance:
        system.attendance.enabled = False

    if args.export_attendance:
        system.attendance.export_attendance(args.export_attendance)
    elif args.web:
        system.run_web(host=args.host, port=args.port)
    else:
        system.run(camera_index=args.camera)
