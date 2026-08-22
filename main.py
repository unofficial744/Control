import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import math
import numpy as np
import threading
import queue
import os

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics"))

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[AI] WARNING: face_recognition module not available. Face recognition will be disabled.")
    print("     Install Visual Studio C++ Build Tools then run: pip install face_recognition")
from ultralytics import YOLO
from util import get_angle, get_distance
from audio import listen_for_commands

# --- 1. Set up Command Queue for Voice ---
command_queue = queue.Queue()
voice_thread = threading.Thread(target=listen_for_commands, args=(command_queue,), daemon=True)
voice_thread.start()

# --- 2. Set up Object & Face Detection AI ---
print("\n[AI] Loading YOLOv8 Object Detection Model... Please wait.")
object_model = YOLO("yolov8n.pt") # Downloads a small, fast AI model on first run

known_face_encodings = []
known_face_names = []
my_image_path = "my_face.jpg"

if FACE_REC_AVAILABLE:
    print("[AI] Face Recognition module found.")
    if os.path.exists(my_image_path):
        print(f"[AI] Learning face from '{my_image_path}'...")
        image = face_recognition.load_image_file(my_image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append("Boss") # <-- Change this to your name
    else:
        print(f"[AI] NOTE: '{my_image_path}' not found. Faces will be detected but labeled as 'Unknown'.")
else:
    print("[AI] Face Recognition disabled (module not installed).")

process_this_frame = True
face_locations = []
face_encodings = []
face_names = []

# --- 3. Set up Hand Tracking ---
HAND_CONNECTIONS = [
    (0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20)
]

def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_lm = hand_landmarks[start_idx]
        end_lm = hand_landmarks[end_idx]
        start_pt = (int(start_lm.x * w), int(start_lm.y * h))
        end_pt = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
    for lm in hand_landmarks:
        pt = (int(lm.x * w), int(lm.y * h))
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

latest_result = None
result_lock = threading.Lock()

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=result_callback
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frameR = 120

click_cooldown = 0.5
freeze_cursor = False
is_dragging = False
zoom_cooldown = 0.1
last_zoom_time = 0
last_media_time = 0
media_cooldown = 1.5
smoothing = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

screen_w, screen_h = pyautogui.size()

# Canvas Variables
canvas_mode = False
drawing_canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

print("\n[System] All systems are running!")
print("[Voice] Say 'canvas' for drawing mode, 'switch off canvas' to close it.")
print("Press 'q' in the video window to quit.")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_timestamp = 0

while True:
    # --- 4. Check for Voice Commands ---
    try:
        cmd = command_queue.get_nowait().lower()
        if "switch off canvas" in cmd:
            canvas_mode = False
            try:
                cv2.destroyWindow('Canvas')
            except cv2.error:
                pass
            print("Canvas Mode: OFF")

        elif "canvas" in cmd:
            canvas_mode = True
            drawing_canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
            cv2.namedWindow('Canvas', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Canvas Mode: ON")
    except queue.Empty:
        pass

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 1
    hand_landmarker.detect_async(mp_image, frame_timestamp)

    # ==========================================
    # 5. OBJECT & FACE DETECTION (Visuals)
    # ==========================================
    # YOLO Object Detection
    results_yolo = object_model(frame, stream=True, verbose=False)
    for r in results_yolo:
        for box in r.boxes:
            cls = int(box.cls[0])
            object_name = object_model.names[cls]
            # Ignore people so our Face Recognizer can handle them
            if object_name != "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, object_name, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Face Recognition (Running on scaled-down image to save CPU)
    small_rgb = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

    if FACE_REC_AVAILABLE:
        if process_this_frame:
            face_locations = face_recognition.face_locations(small_rgb)
            face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if len(known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match = np.argmin(distances)
                    if matches[best_match]:
                        name = known_face_names[best_match]
                face_names.append(name)

        process_this_frame = not process_this_frame # Skip next frame for better performance

        # Draw Face UI
        cv2.putText(frame, f"Faces Detected: {len(face_locations)}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Face Recognition: Disabled", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 200, 200), 2)

    # ==========================================
    # 6. HAND TRACKING & MOUSE LOGIC
    # ==========================================
    with result_lock:
        results = latest_result

    cv2.rectangle(frame, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)

    if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            thumb_tip = hand_landmarks[4]
            thumb_ip = hand_landmarks[3]
            index_mcp = hand_landmarks[5]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            wrist = hand_landmarks[0]
            middle_base = hand_landmarks[9]

            hand_size = math.hypot(middle_base.x - wrist.x, middle_base.y - wrist.y)

            fingers = [
                1 if hand_landmarks[tip].y < hand_landmarks[tip-2].y else 0
                for tip in [8, 12, 16, 20]
            ]
            fingers_up_count = sum(fingers)

            is_thumbs_up = (fingers_up_count == 0) and (thumb_tip.y < thumb_ip.y) and (thumb_tip.y < index_mcp.y)
            is_fist = (fingers_up_count == 0) and not is_thumbs_up

            thumb_to_index_dist = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
            is_thumb_open = hand_size > 0 and (thumb_to_index_dist / hand_size) > 0.6

            is_open_palm = (fingers_up_count == 4) and is_thumb_open
            is_scroll_mode = (fingers_up_count == 4) and not is_thumb_open

            index_x_cam = index_tip.x * cam_w
            index_y_cam = index_tip.y * cam_h

            raw_x = np.interp(index_x_cam, (frameR, cam_w - frameR), (0, screen_w))
            raw_y = np.interp(index_y_cam, (frameR, cam_h - frameR), (0, screen_h))

            clocX = plocX + (raw_x - plocX) / smoothing
            clocY = plocY + (raw_y - plocY) / smoothing

            index_pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            middle_pinch_dist = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)

            if canvas_mode:
                if fingers_up_count > 0:
                    pyautogui.moveTo(clocX, clocY)
                
                # ERASER
                if is_open_palm:
                    cv2.line(drawing_canvas, (int(plocX), int(plocY)), (int(clocX), int(clocY)), (255, 255, 255), 100)
                    cv2.putText(frame, "Erasing...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # DRAW
                elif index_pinch_dist < 0.04 and fingers_up_count > 0:
                    cv2.line(drawing_canvas, (int(plocX), int(plocY)), (int(clocX), int(clocY)), (0, 0, 255), 8)
                    cv2.putText(frame, "Drawing...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Canvas', drawing_canvas)

            else:
                if middle_pinch_dist < 0.04 and fingers_up_count > 0:
                    if not freeze_cursor:
                        freeze_cursor = True
                        pyautogui.doubleClick()
                        cv2.putText(frame, "Double Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                elif index_pinch_dist < 0.04 and fingers_up_count > 0:
                    if not freeze_cursor:
                        freeze_cursor = True
                        pyautogui.click()
                        cv2.putText(frame, "Single Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    if freeze_cursor:
                        time.sleep(0.1)
                    freeze_cursor = False

                if not freeze_cursor and not is_dragging and fingers_up_count > 0:
                    pyautogui.moveTo(clocX, clocY)

                if is_fist:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                    pyautogui.moveTo(clocX, clocY)
                    cv2.putText(frame, "Grabbing...", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                        cv2.putText(frame, "Released", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if is_scroll_mode:
                    if index_tip.y < 0.4:
                        pyautogui.scroll(60)
                    elif index_tip.y > 0.6:
                        pyautogui.scroll(-60)

                current_time = time.time()
                if (is_thumbs_up or is_open_palm) and (current_time - last_media_time >= media_cooldown):
                    pyautogui.press('playpause')
                    last_media_time = current_time

                if is_thumbs_up:
                    cv2.putText(frame, "PLAY (Thumbs Up)", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
                elif is_open_palm:
                    cv2.putText(frame, "PAUSE (Open Palm)", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

                if fingers == [1, 1, 0, 0] and not is_dragging:
                    zoom_dist = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
                    if current_time - last_zoom_time >= zoom_cooldown and hand_size > 0:
                        zoom_ratio = zoom_dist / hand_size
                        if zoom_ratio > 0.6:
                            pyautogui.keyDown('ctrl')
                            pyautogui.scroll(50)
                            pyautogui.keyUp('ctrl')
                            last_zoom_time = current_time
                        elif zoom_ratio < 0.25:
                            pyautogui.keyDown('ctrl')
                            pyautogui.scroll(-50)
                            pyautogui.keyUp('ctrl')
                            last_zoom_time = current_time

            plocX, plocY = clocX, clocY

    cv2.imshow('live video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()