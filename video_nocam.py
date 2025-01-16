import os
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

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

# Function to process and display the video in real-time
def process_video(input_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter to save the output video
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_video_path = os.path.join(output_folder, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
            cv2.putText(frame, "ALERT: Harmful Object Detected with Person!", (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame in real-time
        cv2.imshow('Processed Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define input video path
input_video_path = 'input_video_2.mp4'

# Process and play video in real-time
process_video(input_video_path)

print(f"Processed video saved to {os.path.join('output', 'output_video.mp4')}")
