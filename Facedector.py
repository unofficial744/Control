import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import os

print("Loading Object Detection Model (YOLOv8)... This might take a few seconds on the first run.")
# This will automatically download a tiny, fast 6MB AI model the first time you run it.
object_model = YOLO("yolov8n.pt") 

# --- SETUP KNOWN FACES ---
known_face_encodings = []
known_face_names = []

# To recognize YOUR face, put a picture of yourself in the same folder named "my_face.jpg"
my_image_path = "my_face.jpg"

if os.path.exists(my_image_path):
    print(f"Learning face from {my_image_path}...")
    image = face_recognition.load_image_file(my_image_path)
    # Get the 128-dimension face encoding for this image
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("Boss") # <--- Change this to your name
else:
    print(f"WARNING: '{my_image_path}' not found. I will detect faces, but I won't know their names.")
    print("Add a clear picture of your face named 'my_face.jpg' to this folder to use recognition.")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Process every other frame to save computer power and keep the video smooth
process_this_frame = True
face_locations = []
face_encodings = []
face_names = []

print("\n[Vision] Face and Object detector is running! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # ==========================================
    # 1. OBJECT DETECTION (Laptops, Phones, etc.)
    # ==========================================
    # Run YOLO object detection on the frame (verbose=False hides console spam)
    results = object_model(frame, stream=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the class name of the object (e.g., "cell phone", "laptop")
            cls = int(box.cls[0])
            object_name = object_model.names[cls]
            
            # We skip 'person' because our Face Detector will handle drawing boxes for people
            if object_name != "person":
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                
                # Draw a blue box around objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{object_name} {confidence}", (x1, max(35, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ==========================================
    # 2. FACE DETECTION & RECOGNITION
    # ==========================================
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Default name if we don't recognize them
            name = "Unknown"

            if len(known_face_encodings) > 0:
                # Compare the detected face with our known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                # Find the closest match
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display Face Count on the screen
    face_count = len(face_locations)
    cv2.putText(frame, f"Faces Detected: {face_count}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

    # Draw the boxes and names for faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Pick a color: Green for known people, Red for unknown
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # Show the final image
    cv2.imshow('Face & Object Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()