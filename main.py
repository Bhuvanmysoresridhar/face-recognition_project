"""
Production-Grade Face Recognition System
Demonstrates the complete pipeline beyond basic ML
"""

import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import json

class ProductionFaceRecognition:
    def __init__(self, known_faces_dir="known_faces", 
                 threshold=0.6, min_face_size=50):
        """
        Initialize face recognition system with production features
        
        Args:
            known_faces_dir: Directory with reference images
            threshold: Distance threshold for matching (lower = stricter)
            min_face_size: Minimum face size in pixels
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.min_face_size = min_face_size
        
        # Performance metrics
        self.detection_log = []
        
        # Load known faces
        self.load_known_faces()
    
    def check_image_quality(self, image):
        """
        Quality checks before processing
        
        Returns: (is_good, reason)
        """
        # Check 1: Image too small
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return False, "Image too small"
        
        # Check 2: Blur detection using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur
            return False, "Image too blurry"
        
        # Check 3: Brightness check
        brightness = np.mean(gray)
        if brightness < 50:
            return False, "Image too dark"
        if brightness > 200:
            return False, "Image too bright"
        
        return True, "OK"
    
    def align_face(self, image, face_location):
        """
        Align face for better recognition
        Uses facial landmarks to normalize pose
        
        Args:
            image: Original image
            face_location: (top, right, bottom, left) tuple
            
        Returns: Aligned face image
        """
        # Get facial landmarks
        face_landmarks = face_recognition.face_landmarks(image, [face_location])
        
        if not face_landmarks:
            return image
        
        landmarks = face_landmarks[0]
        
        # Get eye positions
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        
        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                       (left_eye[1] + right_eye[1]) / 2)
        
        # Rotate image to align eyes horizontally
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, 
                                 (image.shape[1], image.shape[0]))
        
        return aligned
    
    def load_known_faces(self):
        """
        Load known faces with quality checks and augmentation
        """
        print("Loading known faces with quality checks...")
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory: {self.known_faces_dir}")
            return
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.known_faces_dir, filename)
                image = cv2.imread(image_path)
                
                # Quality check
                is_good, reason = self.check_image_quality(image)
                if not is_good:
                    print(f"Skipping {filename}: {reason}")
                    continue
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face
                face_locations = face_recognition.face_locations(rgb_image)
                
                if len(face_locations) == 0:
                    print(f"No face found in {filename}")
                    continue
                
                if len(face_locations) > 1:
                    print(f"Multiple faces in {filename}, using first one")
                
                # Get face encoding
                face_location = face_locations[0]
                
                # Align face for better encoding
                aligned = self.align_face(rgb_image, face_location)
                
                # Generate encoding from aligned face
                encodings = face_recognition.face_encodings(aligned, [face_location])
                
                if len(encodings) > 0:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]
                    
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"✓ Loaded: {name}")
        
        print(f"Loaded {len(self.known_face_names)} faces successfully")
    
    def recognize_face(self, face_encoding):
        """
        Recognize face using distance-based matching
        
        Returns: (name, confidence, distance)
        """
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0.0, 1.0
        
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]
        
        # Check if below threshold
        if min_distance < self.threshold:
            name = self.known_face_names[best_match_index]
            confidence = 1 - min_distance  # Convert to confidence score
            return name, confidence, min_distance
        else:
            return "Unknown", 0.0, min_distance
    
    def log_detection(self, name, confidence, timestamp):
        """
        Log detection for analytics
        """
        log_entry = {
            "name": name,
            "confidence": confidence,
            "timestamp": timestamp.isoformat()
        }
        self.detection_log.append(log_entry)
    
    def save_logs(self, filename="detection_log.json"):
        """
        Save detection logs to file
        """
        with open(filename, 'w') as f:
            json.dump(self.detection_log, f, indent=2)
        print(f"Logs saved to {filename}")
    
    def run(self, camera_index=0):
        """
        Main recognition loop with all production features
        
        Args:
            camera_index: Camera index to use (default: 0)
        """
        # Try multiple backends for better compatibility
        backends = [
            cv2.CAP_ANY,           # Auto-detect
            cv2.CAP_DSHOW,         # DirectShow (Windows)
            cv2.CAP_AVFOUNDATION,  # AVFoundation (macOS)
            cv2.CAP_V4L2,          # Video4Linux (Linux)
        ]
        
        video_capture = None
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret:
                        video_capture = cap
                        print(f"✅ Camera opened successfully")
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                continue
        
        if video_capture is None or not video_capture.isOpened():
            print("\n❌ ERROR: Could not open webcam!")
            print("\nTroubleshooting steps:")
            print("1. Run 'python webcam_test.py' to diagnose the issue")
            print("2. Close other apps using the camera (Zoom, Teams, etc.)")
            print("3. Check camera permissions in your OS settings")
            print("4. Try different camera indices: system.run(camera_index=1)")
            return
        
        print("\n=== Face Recognition System Started ===")
        print(f"Threshold: {self.threshold}")
        print(f"Known faces: {len(self.known_face_names)}")
        print("Press 'q' to quit, 's' to save logs")
        print("=" * 40 + "\n")
        
        process_this_frame = True
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            if process_this_frame:
                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                # Filter by minimum size
                valid_faces = []
                for (top, right, bottom, left) in face_locations:
                    face_width = (right - left) * 4
                    face_height = (bottom - top) * 4
                    if face_width >= self.min_face_size and face_height >= self.min_face_size:
                        valid_faces.append((top, right, bottom, left))
                
                face_locations = valid_faces
                
                # Generate encodings
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )
                
                face_names = []
                face_confidences = []
                
                for face_encoding in face_encodings:
                    name, confidence, distance = self.recognize_face(face_encoding)
                    face_names.append(name)
                    face_confidences.append(confidence)
                    
                    # Log recognized faces (not "Unknown")
                    if name != "Unknown":
                        self.log_detection(name, confidence, datetime.now())
            
            process_this_frame = not process_this_frame
            
            # Display results
            for (top, right, bottom, left), name, conf in zip(
                face_locations, face_names, face_confidences
            ):
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Choose color based on recognition
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Prepare label
                if name != "Unknown":
                    label = f"{name} ({conf:.2f})"
                else:
                    label = "Unknown"
                
                # Draw label background
                cv2.rectangle(frame, (left, bottom - 35), 
                            (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Add info overlay
            info_text = f"Faces: {len(face_locations)} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_logs()
                print("Logs saved!")
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n=== Session Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections logged: {len(self.detection_log)}")
        
        if self.detection_log:
            unique_people = set(log['name'] for log in self.detection_log)
            print(f"Unique people detected: {len(unique_people)}")
            print(f"People: {', '.join(unique_people)}")


if __name__ == "__main__":
    # Initialize with custom settings
    system = ProductionFaceRecognition(
        known_faces_dir="known_faces",
        threshold=0.6,        # Adjust for stricter/looser matching
        min_face_size=50      # Minimum face size in pixels
    )
    
    # Run the system with camera index 1
    system.run(camera_index=1)