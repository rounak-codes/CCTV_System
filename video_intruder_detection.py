import os
import cv2
import numpy as np
from flask import Flask, Response, render_template
from ultralytics import YOLO
from datetime import datetime

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

app = Flask(__name__)

# Load the YOLO model
yolo_model = YOLO("yolo11x.pt")
custom_model = YOLO("yolov8s-worldv2.pt")

# Define harmful object classes for YOLO and the custom model
yolo_classes = {
    0: "person",
    34: "baseball bat",
    35: "baseball glove",
    39: "bottle",
    42: "fork",
    43: "knife",
    76: "scissors",
}

# Add offset to custom model classes to avoid collision with YOLO classes
custom_classes_offset = 100  # Offset custom model class IDs by 100
custom_classes = {
    0 + custom_classes_offset: "handgun",
    1 + custom_classes_offset: "knife",
    2 + custom_classes_offset: "rifle",
}

# Combine both YOLO and custom model classes
combined_classes = {**yolo_classes, **custom_classes}

# Global list to store camera indices
camera_indexes = []

# Function to find available camera indices
def get_camera_indices():
    available_cameras = []
    for i in range(10):  # Try indices from 0 to 9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Initialize camera connections
def initialize_cameras():
    global camera_indexes

    # Get the list of available camera indices
    camera_indexes = get_camera_indices()

    if len(camera_indexes) < 1:
        print("Not enough cameras available.")
        return None

    # Initialize VideoCapture objects for available cameras
    cameras = []
    for index in camera_indexes:  # Initialize all detected cameras, even if it's just one
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Failed to open camera at index {index}.")
            return None
        cameras.append(cap)
    
    return cameras

# Function to reconnect to a disconnected camera
def reconnect_camera(camera_id):
    global camera_indexes

    if len(camera_indexes) <= camera_id:
        print(f"Camera {camera_id} is no longer available.")
        return None

    # Reinitialize the camera
    return cv2.VideoCapture(camera_indexes[camera_id])

# Round the timestamp to the nearest minute
def get_rounded_timestamp():
    now = datetime.now()
    return now.replace(second=0, microsecond=0)

# Frame generation for video feed
def generate_frames(camera_id, cameras):
    last_logged_minute = None  # To track the last logged minute
    while True:
        ret, frame = cameras[camera_id].read()

        if not ret:
            # If the frame can't be read, reconnect to the camera
            print(f"Camera {camera_id} is disconnected. Attempting reconnection...")
            cameras[camera_id] = reconnect_camera(camera_id)
            if cameras[camera_id] is None:
                # If reconnection fails, return an error response
                frame = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)
                cv2.putText(frame, "No feed available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

        # Perform object detection (YOLO and Custom Model)
        results_yolo = yolo_model(frame)
        results_custom = custom_model(frame)

        person_detected = False
        harmful_objects_detected = []

        # Process YOLO results
        for result in results_yolo:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                class_id = int(box.cls[0])             # Object class ID
                confidence = box.conf[0] * 100         # Confidence score

                if class_id == 0:  # Person class in YOLO
                    person_detected = True
                elif class_id in yolo_classes and class_id != 0:  # Harmful object
                    harmful_objects_detected.append((x1, y1, x2, y2, yolo_classes[class_id], confidence))

        # Process custom model results
        for result in results_custom:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                class_id = int(box.cls[0])             # Object class ID
                confidence = box.conf[0] * 100         # Confidence score

                if class_id == 0 + custom_classes_offset:  # Handgun
                    harmful_objects_detected.append((x1, y1, x2, y2, custom_classes[class_id], confidence))
                elif class_id == 1 + custom_classes_offset:  # Knife
                    harmful_objects_detected.append((x1, y1, x2, y2, custom_classes[class_id], confidence))
                elif class_id == 2 + custom_classes_offset:  # Rifle
                    harmful_objects_detected.append((x1, y1, x2, y2, custom_classes[class_id], confidence))

        # If both a person and a harmful object are detected, trigger an alert
        if person_detected and harmful_objects_detected:
            for obj in harmful_objects_detected:
                x1, y1, x2, y2, label, confidence = obj
                # Draw rectangle and label for harmful objects
                color = (0, 0, 255)  # Red for harmful objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Visual alert for detection
            alert_message = f"ALERT!! (Camera {camera_id})"
            cv2.putText(frame, alert_message, (100, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 3)

            # Log the timestamp for detection, rounding to the nearest minute
            current_minute = get_rounded_timestamp()
            if current_minute != last_logged_minute:
                last_logged_minute = current_minute
                with open("detection_log.txt", "a") as log_file:
                    log_file.write(f"{current_minute} - Harmful object detected with person (Camera {camera_id})\n")

        # Get the current time for overlay
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Overlay the current time on the frame
        cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    # Initialize cameras on the first request
    global cameras
    if not hasattr(video_feed, "cameras"):
        cameras = initialize_cameras()
        if cameras is None:
            return "Error: Unable to initialize cameras", 500

    # Ensure the camera_id is valid
    if camera_id >= len(cameras):
        return "Error: Camera ID out of range", 400

    return Response(generate_frames(camera_id, cameras), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
