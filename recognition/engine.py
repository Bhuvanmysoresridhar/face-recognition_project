"""
Core face recognition engine with multi-image support, face alignment,
quality checks, and optional FAISS-based fast nearest-neighbor indexing.
"""

import os
import numpy as np
import cv2
import face_recognition

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaceRecognitionEngine:
    """
    Enhanced recognition engine supporting:
    - Multiple reference images per person
    - Persistent encoding cache
    - FAISS indexing for large face databases
    - Image quality validation and face alignment
    """

    def __init__(self, config, encoding_cache=None, database=None):
        self.config = config
        self.cache = encoding_cache
        self.db = database

        rec = config.section("recognition")
        self.threshold = rec.get("threshold", 0.6)
        self.min_face_size = rec.get("min_face_size", 50)
        self.model = rec.get("model", "hog")
        self.frame_scale = rec.get("frame_scale", 0.25)
        self.skip_frames = rec.get("skip_frames", 2)

        self.known_faces_dir = config.get("paths", "known_faces_dir", default="known_faces")

        # Storage
        self.known_encodings = []   # flat list of encodings
        self.known_names = []       # parallel list of names
        self.person_encodings = {}  # name -> [encodings]

        self._faiss_index = None

        self.load_known_faces()

    # ------------------------------------------------------------------ quality

    @staticmethod
    def check_image_quality(image):
        """Validate image quality. Returns (is_good, reason)."""
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False, "Image too small"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
            return False, "Image too blurry"
        brightness = np.mean(gray)
        if brightness < 50:
            return False, "Image too dark"
        if brightness > 200:
            return False, "Image too bright"
        return True, "OK"

    # ----------------------------------------------------------------- alignment

    @staticmethod
    def align_face(image, face_location):
        """Align face using eye landmarks for consistent encoding."""
        landmarks = face_recognition.face_landmarks(image, [face_location])
        if not landmarks:
            return image
        lm = landmarks[0]
        left_eye = np.mean(lm["left_eye"], axis=0)
        right_eye = np.mean(lm["right_eye"], axis=0)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # ------------------------------------------------------------- loading faces

    def _encode_image(self, image_path):
        """Load a single image, validate, align, and return encodings list."""
        image = cv2.imread(image_path)
        if image is None:
            return []
        ok, reason = self.check_image_quality(image)
        if not ok:
            print(f"  Skipping {os.path.basename(image_path)}: {reason}")
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model=self.model)
        if not locations:
            print(f"  No face in {os.path.basename(image_path)}")
            return []
        aligned = self.align_face(rgb, locations[0])
        encs = face_recognition.face_encodings(aligned, [locations[0]])
        return list(encs)

    def load_known_faces(self):
        """
        Load faces from known_faces_dir.
        Supports both flat layout (name.jpg) and folder layout (name/*.jpg).
        Uses encoding cache when available.
        """
        print("Loading known faces...")
        self.known_encodings.clear()
        self.known_names.clear()
        self.person_encodings.clear()

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
            print(f"Created directory: {self.known_faces_dir}")
            return

        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        # Collect person -> [image_paths]
        persons = {}
        for entry in sorted(os.listdir(self.known_faces_dir)):
            full = os.path.join(self.known_faces_dir, entry)
            if os.path.isdir(full):
                # Folder layout: known_faces/john/*.jpg
                imgs = [
                    os.path.join(full, f)
                    for f in sorted(os.listdir(full))
                    if f.lower().endswith(image_exts)
                ]
                if imgs:
                    persons[entry] = imgs
            elif entry.lower().endswith(image_exts):
                # Flat layout: known_faces/john.jpg
                name = os.path.splitext(entry)[0]
                persons.setdefault(name, []).append(full)

        for name, paths in persons.items():
            encodings = []

            # Try cache first
            if self.cache:
                cached, needs_update = self.cache.get_encodings(name, paths)
                if not needs_update and cached:
                    encodings = cached
                    print(f"  [cache] {name}: {len(encodings)} encoding(s)")
                else:
                    for p in paths:
                        encodings.extend(self._encode_image(p))
                    if encodings:
                        self.cache.store_encodings(name, paths, encodings)
                    print(f"  [new]   {name}: {len(encodings)} encoding(s)")
            else:
                for p in paths:
                    encodings.extend(self._encode_image(p))
                print(f"  {name}: {len(encodings)} encoding(s)")

            if encodings:
                self.person_encodings[name] = encodings
                for enc in encodings:
                    self.known_encodings.append(enc)
                    self.known_names.append(name)

            # Update DB person record
            if self.db:
                self.db.add_person(name)
                self.db.update_image_count(name, len(paths))

        self._build_index()
        print(f"Loaded {len(self.person_encodings)} people, "
              f"{len(self.known_encodings)} total encodings")

    def _build_index(self):
        """Build FAISS index for fast nearest-neighbor search if available."""
        if not HAS_FAISS or len(self.known_encodings) == 0:
            self._faiss_index = None
            return
        dim = 128  # face_recognition uses 128-d vectors
        index = faiss.IndexFlatL2(dim)
        matrix = np.array(self.known_encodings, dtype=np.float32)
        index.add(matrix)
        self._faiss_index = index
        print(f"  FAISS index built with {index.ntotal} vectors")

    # ------------------------------------------------------------- recognition

    def recognize_face(self, face_encoding):
        """
        Match a face encoding against known faces.
        Returns (name, confidence, distance).
        """
        if not self.known_encodings:
            return "Unknown", 0.0, 1.0

        if self._faiss_index is not None:
            query = np.array([face_encoding], dtype=np.float32)
            distances, indices = self._faiss_index.search(query, 1)
            # FAISS L2 returns squared distance, face_recognition uses euclidean
            min_distance = float(np.sqrt(distances[0][0]))
            best_idx = int(indices[0][0])
        else:
            face_distances = face_recognition.face_distance(
                self.known_encodings, face_encoding
            )
            best_idx = int(np.argmin(face_distances))
            min_distance = float(face_distances[best_idx])

        if min_distance < self.threshold:
            name = self.known_names[best_idx]
            confidence = 1.0 - min_distance
            return name, confidence, min_distance
        return "Unknown", 0.0, min_distance

    # -------------------------------------------------------- webcam registration

    def register_face_from_frame(self, frame, name):
        """
        Register a new face directly from a webcam frame.
        Saves the image and updates encodings.
        Returns True on success.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model=self.model)
        if not locations:
            return False

        # Create person directory
        person_dir = os.path.join(self.known_faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        existing = len([f for f in os.listdir(person_dir) if f.endswith((".jpg", ".png"))])
        img_path = os.path.join(person_dir, f"{name}_{existing + 1}.jpg")
        cv2.imwrite(img_path, frame)

        # Encode and add
        aligned = self.align_face(rgb, locations[0])
        encs = face_recognition.face_encodings(aligned, [locations[0]])
        if encs:
            self.known_encodings.append(encs[0])
            self.known_names.append(name)
            self.person_encodings.setdefault(name, []).append(encs[0])
            self._build_index()

            if self.cache:
                all_paths = [
                    os.path.join(person_dir, f)
                    for f in sorted(os.listdir(person_dir))
                    if f.endswith((".jpg", ".png"))
                ]
                self.cache.store_encodings(name, all_paths, self.person_encodings[name])
            if self.db:
                self.db.add_person(name)
                count = len(self.person_encodings[name])
                self.db.update_image_count(name, count)
            print(f"Registered {name} ({len(self.person_encodings[name])} images)")
            return True
        return False
