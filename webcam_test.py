"""
Simple webcam test script to diagnose camera issues
"""

import cv2
import sys

def test_camera(camera_index=0):
    """Test if camera can be opened and used"""
    
    print("=" * 50)
    print("WEBCAM DIAGNOSTIC TEST")
    print("=" * 50)
    
    # List available backends
    backends = {
        cv2.CAP_ANY: "Auto-detect",
        cv2.CAP_DSHOW: "DirectShow (Windows)",
        cv2.CAP_AVFOUNDATION: "AVFoundation (macOS)",
        cv2.CAP_V4L2: "Video4Linux (Linux)",
    }
    
    print(f"\nTesting camera index: {camera_index}")
    print("\nAvailable backends:")
    for backend_id, name in backends.items():
        print(f"  - {name} ({backend_id})")
    
    print("\n" + "=" * 50)
    print("Testing each backend...")
    print("=" * 50)
    
    working_backend = None
    
    for backend_id, backend_name in backends.items():
        try:
            print(f"\nTrying {backend_name}...")
            cap = cv2.VideoCapture(camera_index, backend_id)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"  ✅ SUCCESS!")
                    print(f"     Resolution: {width}x{height}")
                    print(f"     Backend: {backend_name}")
                    working_backend = (backend_id, backend_name)
                    cap.release()
                    break
                else:
                    print(f"  ⚠️  Opened but cannot read frames")
                    cap.release()
            else:
                print(f"  ❌ Cannot open camera")
                cap.release()
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if working_backend:
        print("\n" + "=" * 50)
        print("✅ CAMERA WORKING!")
        print("=" * 50)
        print(f"Use backend: {working_backend[1]} ({working_backend[0]})")
        print(f"Camera index: {camera_index}")
        return True
    else:
        print("\n" + "=" * 50)
        print("❌ CAMERA NOT WORKING")
        print("=" * 50)
        print("\nTroubleshooting:")
        print("1. Check camera permissions in System Settings")
        print("2. Close other apps using the camera (Zoom, Teams, etc.)")
        print("3. Try a different camera index:")
        print("   python webcam_test.py 1")
        print("   python webcam_test.py 2")
        print("4. Restart your computer")
        return False

def test_multiple_cameras():
    """Test multiple camera indices"""
    print("\n" + "=" * 50)
    print("TESTING MULTIPLE CAMERA INDICES")
    print("=" * 50)
    
    for i in range(5):
        print(f"\n--- Camera {i} ---")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {i} is available")
                cap.release()
            else:
                print(f"⚠️  Camera {i} opened but cannot read")
                cap.release()
        else:
            print(f"❌ Camera {i} not available")
            cap.release()

if __name__ == "__main__":
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("Invalid camera index. Using default (0)")
    
    # Test the specified camera
    success = test_camera(camera_index)
    
    # If failed, test other cameras
    if not success:
        test_multiple_cameras()

