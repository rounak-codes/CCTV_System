import os
import signal
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, send_from_directory
from ultralytics import YOLO
from datetime import datetime
from time import sleep
import subprocess
import threading


os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

app = Flask(__name__)

# Load the YOLO model
yolo_model = YOLO("yolo11s.pt")
custom_model = YOLO("violenceprediction.pt")

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
    0 + custom_classes_offset: "Grenade",
    1 + custom_classes_offset: "Knife",
    2 + custom_classes_offset: "Missile",
    3 + custom_classes_offset: "Pistol",
    4 + custom_classes_offset: "Rifle",
    5 + custom_classes_offset: "armed man",
    6 + custom_classes_offset: "body",
    7 + custom_classes_offset: "face",
    8 + custom_classes_offset: "hand",
}

# Combine both YOLO and custom model classes
combined_classes = {**yolo_classes, **custom_classes}

cameras = []

# Global list to store camera indices
#camera_indexes = []

# Function to find available camera indices
def get_camera_indices():
    available_cameras = []
    for i in range(3):  # Check the first 3 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Initialize camera connections
def initialize_cameras():
    global cameras
    camera_indexes = get_camera_indices()

    if not camera_indexes:
        print("No cameras available.")
        return None

    # Initialize VideoCapture objects for available cameras
    for index in camera_indexes:  # Initialize all detected cameras
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Failed to open camera at index {index}.")
            continue
        cameras.append(cap)
    
    return cameras

# Function to reconnect to a disconnected camera
def reconnect_camera(camera_id):
    if camera_id >= len(cameras):
        print(f"Camera {camera_id} is no longer available.")
        return None
    return cv2.VideoCapture(camera_id)

# Round the timestamp to the nearest minute
def get_rounded_timestamp():
    now = datetime.now()
    return now.replace(second=0, microsecond=0)


def generate_frames(camera_id):
    if camera_id >= len(cameras) or not cameras[camera_id].isOpened():
        print(f"Camera {camera_id} is not available.")
        return

    process = None

    try:
        # Setup video recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for the video filename
        video_filename = f"static/recordings/camera_{camera_id}_{timestamp}.mp4"
        
        # Define FFmpeg command for encoding
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-f', 'rawvideo',  # Input format
            '-vcodec', 'rawvideo',  # Input codec
            '-pix_fmt', 'bgr24',  # Pixel format (OpenCV uses bgr24)
            '-s', '640x480',  # Resolution (same as the video frame size)
            '-r', '25',  # Frame rate
            '-i', '-',  # Input from stdin (pipe)
            '-c:v', 'libx264',  # H.264 codec for output
            '-preset', 'fast',  # Encoding speed/quality trade-off
            '-crf', '23',  # Constant rate factor for quality (lower is better)
            video_filename  # Output file
        ]
        
        # Open a subprocess for FFmpeg to handle video encoding
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        last_logged_minute = None  # To track the last logged minute
        
        while True:

            ret, frame = cameras[camera_id].read()

            if not ret:
                # If the frame can't be read, reconnect to the camera
                print(f"Camera {camera_id} is disconnected. Attempting reconnection...")
                cameras[camera_id] = reconnect_camera(camera_id)
                if cameras[camera_id] is None:
                    # If reconnection fails, record a blank frame
                    frame = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)
                    process.stdin.write(frame.tobytes())  # Write the blank frame
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

                    if class_id + custom_classes_offset in custom_classes:
                        label = custom_classes[class_id + custom_classes_offset]
                        harmful_objects_detected.append((x1, y1, x2, y2, label, confidence))

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
            cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            process.stdin.write(frame.tobytes())

            # Yield the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        if process:
            process.stdin.close()
            process.wait()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    global cameras

    available_indices = get_camera_indices()
    for index in available_indices:
        if index >= len(cameras) or not cameras[index].isOpened():
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                if index < len(cameras):
                    cameras[index] = cap
                else:
                    cameras.append(cap)

    # Remove any cameras not in current available indices
    cameras[:] = [cam for i, cam in enumerate(cameras) if i in available_indices]

    if camera_id >= len(cameras) or not cameras[camera_id].isOpened():
        return "Error: Camera ID out of range or not available", 400

    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/refresh_cameras', methods=['POST'])
def refresh_cameras():
    global cameras
    for cam in cameras:
        cam.release()
    cameras.clear()
    initialize_cameras()
    return "Cameras refreshed", 200


@app.route('/recordings')
def recordings():
    files = [f for f in os.listdir('static/recordings') if f.endswith('.mp4')]
    files_with_times = [(f, os.path.getmtime(os.path.join('static/recordings', f))) for f in files]
    files_with_times.sort(key=lambda x: x[1], reverse=True)
    files_to_display = [f[0] for f in files_with_times[1:]]
    
    return render_template('recordings.html', files=files_to_display)

# Route to play selected recording
@app.route('/play/<filename>')
def play_recording(filename):
    return render_template('play.html', filename=filename)

# Serve video files from the recordings folder
@app.route('/static/recordings/<filename>')
def serve_recording(filename):
    return send_from_directory('static/recordings', filename)

@app.route('/logs')
def read_logs():
    log_entries = []
    with open("detection_log.txt", "r") as file:
        for line in file:
            # Parse each line into timestamp, message, and camera
            parts = line.strip().split(" - ")
            if len(parts) == 2:
                timestamp, message = parts
                camera = message.split("(")[-1].replace(")", "").strip()  # Extract camera info
                log_message = message.split("(")[0].strip()  # Extract log message
                log_entries.append({
                    "timestamp": timestamp,
                    "log_message": log_message,
                    "camera": camera
                })
    return log_entries


@app.route('/shutdown_server', methods=['POST'])
def shutdown_server():
    # Stop camera feeds
    for camera in cameras:
        camera.release()
    sleep(1)
    os.kill(os.getpid(), signal.SIGINT)   
    return '', 200


if __name__ == "__main__":
    # Create the recordings directory if it doesn't exist
    if not os.path.exists('static/recordings'):
        os.makedirs('static/recordings')

    # Initialize cameras before running the app
    initialize_cameras()
    app.run(debug=True, host="0.0.0.0", port=5000)