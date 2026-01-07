# Help Guide - Face Recognition System

## 📁 Your Files Explained

### `main.py` (Updated)
- **Main application** with improved camera handling
- Tries multiple camera backends automatically
- Production-ready face recognition system
- **Use this for normal operation**

### `test_webcam.py`
- Same as main.py but was used for testing
- You can delete this or keep it as backup

### `webcam_test.py` (New)
- **Diagnostic tool** for camera issues
- Tests different camera backends
- Helps find which camera index works
- **Run this if you have camera problems**

---

## 🚀 Quick Commands

### Run Face Recognition (Normal Use)
```bash
cd "/Users/bhuvanms/Desktop/Face recognition/face-recognition_project"
source ../venv/bin/activate
python main.py
```

### Test Camera (If Having Issues)
```bash
python webcam_test.py
# Or test specific camera:
python webcam_test.py 1
```

---

## 🔧 Common Issues & Solutions

### Issue: "Could not open webcam"

**Solution 1: Run diagnostic**
```bash
python webcam_test.py
```

**Solution 2: Try different camera index**
Edit `main.py` line 331:
```python
system.run(camera_index=1)  # Try 1, 2, 3, etc.
```

**Solution 3: Check permissions**
- macOS: System Settings → Privacy & Security → Camera
- Make sure Terminal/Python has camera access

**Solution 4: Close other apps**
- Close Zoom, Teams, FaceTime, or any app using camera
- Restart terminal/IDE

---

## 🎯 What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `main.py` | Main application | **Normal use** - Run this |
| `webcam_test.py` | Camera diagnostics | Camera not working |
| `test_webcam.py` | Backup/test version | Can delete or keep |
| `requirements.txt` | Dependencies | Install packages |
| `README.md` | Full documentation | Read for details |

---

## 💡 Tips

1. **First time running?**
   - Run `python webcam_test.py` first to verify camera works
   - Then run `python main.py`

2. **Camera still not working?**
   - Try: `system.run(camera_index=1)` in code
   - Check System Settings → Privacy → Camera
   - Restart your computer

3. **Improving recognition?**
   - Use clear, front-facing photos in `known_faces/`
   - Adjust threshold in `main.py` (line 326)
   - Lower threshold = stricter matching

---

## 📊 Status Check

Run this to verify everything:
```bash
python -c "
import cv2, face_recognition, numpy as np
print('✅ All imports OK')
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera available')
    cap.release()
else:
    print('❌ Camera issue - run webcam_test.py')
"
```

---

## 🆘 Still Need Help?

1. **Camera issues?** → Run `python webcam_test.py`
2. **Recognition not working?** → Check `known_faces/` folder has good images
3. **Import errors?** → Run `pip install -r requirements.txt`
4. **Performance slow?** → System processes every other frame (normal)

---

## 🎮 Controls During Runtime

- **'q'** = Quit application
- **'s'** = Save detection logs to `detection_log.json`

---

**Everything is set up and ready!** Just run `python main.py` to start.

