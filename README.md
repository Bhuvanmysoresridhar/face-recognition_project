# Face Recognition System

A production-grade face recognition system with real-time video processing, quality checks, and analytics.

## Features

- ✅ Real-time face recognition from webcam
- ✅ Quality checks (blur, brightness, size)
- ✅ Face alignment for better accuracy
- ✅ Detection logging and analytics
- ✅ Configurable recognition threshold
- ✅ Multiple known faces support

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source ../venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

**Note:** If `dlib` fails to install, you may need to install `cmake` first:
```bash
pip install cmake
# or via Homebrew: brew install cmake
```

### 2. Add Known Faces

Place images of people you want to recognize in the `known_faces/` directory:
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Use the person's name as the filename (e.g., `john.jpg`, `sarah.png`)
- Each image should contain **one clear face**
- Good quality images work best (not blurry, well-lit)

Example:
```
known_faces/
  ├── bhuvan.jpeg
  ├── virat.jpeg
  ├── sampreeth.jpeg
  └── ...
```

### 3. Run the System

```bash
python main.py
```

## Usage

### Controls

- **'q'** - Quit the application
- **'s'** - Save detection logs to `detection_log.json`

### Configuration

Edit `main.py` to adjust settings:

```python
system = ProductionFaceRecognition(
    known_faces_dir="known_faces",
    threshold=0.6,        # Lower = stricter matching (0.4-0.7 recommended)
    min_face_size=50      # Minimum face size in pixels
)
```

**Threshold Guide:**
- `0.4` - Very strict (fewer false positives, more false negatives)
- `0.6` - Balanced (default)
- `0.7` - More lenient (more matches, but may have false positives)

## How It Works

1. **Loading Phase**: Scans `known_faces/` directory and encodes all faces
2. **Quality Checks**: Validates image quality (size, blur, brightness)
3. **Face Alignment**: Aligns faces using eye landmarks for better accuracy
4. **Recognition**: Compares detected faces with known encodings using distance metrics
5. **Display**: Shows bounding boxes and names in real-time

## Output

- **Green box** = Recognized person (with confidence score)
- **Red box** = Unknown person
- Detection logs saved to `detection_log.json` when you press 's'

## Troubleshooting

### "Could not open webcam"
- Check camera permissions in System Settings
- Make sure no other app is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

### "No face found in [filename]"
- Image quality may be too low
- Face may be too small or partially obscured
- Try a clearer, front-facing photo

### Poor Recognition Accuracy
- Use higher quality reference images
- Ensure faces are well-lit and front-facing
- Lower the threshold (e.g., 0.5) for stricter matching
- Add multiple images of the same person from different angles

### Performance Issues
- The system processes every other frame by default
- Reduce video resolution in code if needed
- Close other applications to free up resources

## Project Structure

```
face-recognition_project/
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── known_faces/            # Reference images directory
├── detection_log.json      # Generated detection logs
└── README.md              # This file
```

## Dependencies

- `opencv-python` - Video capture and image processing
- `face-recognition` - Face detection and encoding
- `numpy` - Numerical operations
- `dlib` - Machine learning backend (auto-installed with face-recognition)

## License

This project is for educational purposes.

