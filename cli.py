"""
CLI interface with argparse for full configuration control.
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Face Recognition System - Production-grade real-time face recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run with defaults from config.yaml
  python main.py --camera 1 --threshold 0.5   # Use camera 1 with stricter matching
  python main.py --web --port 8080            # Start web dashboard on port 8080
  python main.py --no-liveness --no-tracking  # Disable liveness and tracking
  python main.py --model cnn                  # Use CNN model (GPU required)
        """,
    )

    # Core settings
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device index (default: from config)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Recognition threshold 0.0-1.0 (lower=stricter, default: 0.6)",
    )
    parser.add_argument(
        "--min-face-size", type=int, default=None,
        help="Minimum face size in pixels (default: 50)",
    )
    parser.add_argument(
        "--model", choices=["hog", "cnn"], default=None,
        help="Detection model: hog (CPU) or cnn (GPU)",
    )

    # Feature toggles
    parser.add_argument(
        "--no-liveness", action="store_true",
        help="Disable anti-spoofing/liveness detection",
    )
    parser.add_argument(
        "--no-tracking", action="store_true",
        help="Disable face tracking between frames",
    )
    parser.add_argument(
        "--no-attendance", action="store_true",
        help="Disable attendance tracking",
    )

    # Web dashboard
    parser.add_argument(
        "--web", action="store_true",
        help="Start Flask web dashboard instead of OpenCV window",
    )
    parser.add_argument(
        "--host", default=None,
        help="Web dashboard host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Web dashboard port (default: 5000)",
    )

    # Export
    parser.add_argument(
        "--export-attendance", metavar="DATE",
        help="Export attendance for a date (YYYY-MM-DD) and exit",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Handle export-only mode
    if args.export_attendance:
        from utils.config import Config
        from utils.database import Database
        from utils.attendance import AttendanceManager

        config = Config(args.config)
        db = Database(config.get("paths", "database"))
        att = AttendanceManager(config, db)
        att.export_attendance(args.export_attendance)
        db.close()
    else:
        from main import FaceRecognitionSystem

        system = FaceRecognitionSystem(config_path=args.config)

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

        if args.web:
            system.run_web(host=args.host, port=args.port)
        else:
            system.run(camera_index=args.camera)
