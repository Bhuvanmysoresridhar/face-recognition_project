# Quick Start Guide

## 🚀 Run in 3 Steps

### Step 1: Activate Virtual Environment
```bash
cd "/Users/bhuvanms/Desktop/Face recognition/face-recognition_project"
source ../venv/bin/activate
```

### Step 2: Run the Application
```bash
python main.py
```

### Step 3: Use Controls
- **'q'** = Quit
- **'s'** = Save logs

---

## 📋 What You Have

✅ **5 Known Faces Ready:**
- bhuvan.jpeg
- prathap.jpeg
- sampreeth.jpeg
- sanjith.jpeg
- virat.jpeg

✅ **All Dependencies Installed:**
- opencv-python ✓
- face-recognition ✓
- numpy ✓
- dlib ✓
- setuptools ✓

---

## 🎯 Common Tasks

### Add a New Person
1. Take a clear photo of the person
2. Save it in `known_faces/` folder
3. Name it with their name (e.g., `john.jpg`)
4. Restart the application

### Improve Recognition
- Lower threshold (0.5) = stricter matching
- Higher threshold (0.7) = more lenient
- Use clear, front-facing photos
- Good lighting helps

### Fix Issues

**Camera not opening?**
- Check System Settings → Privacy → Camera
- Close other apps using camera
- Try: `cv2.VideoCapture(1)` instead of `0` in code

**Not recognizing someone?**
- Use a better quality reference image
- Ensure face is clearly visible
- Try multiple angles of the same person

---

## 📊 Understanding the Output

- **Green box** = Recognized (name + confidence %)
- **Red box** = Unknown person
- **Top-left text** = Frame count and face count

---

## 🔧 Customization

Edit `main.py` around line 324:

```python
system = ProductionFaceRecognition(
    known_faces_dir="known_faces",
    threshold=0.6,        # Adjust this (0.4-0.7)
    min_face_size=50       # Minimum face size
)
```

---

## 📁 Project Files

- `main.py` - Main application code
- `requirements.txt` - Dependencies
- `known_faces/` - Your reference images
- `detection_log.json` - Generated logs (when you press 's')
- `README.md` - Full documentation

---

## ❓ Still Need Help?

Check `README.md` for detailed troubleshooting and advanced features.

