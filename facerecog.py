import cv2
import face_recognition
import numpy as np
import os

# Load the known face encodings and their names
def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    
    # Iterate through all images in the directory
    for filename in os.listdir(known_faces_dir):
        file_path = os.path.join(known_faces_dir, filename)
        
        # Load and encode the image
        image = face_recognition.load_image_file(file_path)
        encoding = face_recognition.face_encodings(image)[0]
        
        # Add the encoding and name to the list
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0])  # Use filename (without extension) as the name
    
    return known_encodings, known_names

# Real-time face detection and recognition
# Real-time face detection and recognition
def recognize_faces(known_encodings, known_names):
    # Open a connection to the webcam
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Capture a single frame from the webcam
        ret, frame = video_capture.read()
        
        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
        
        # Detect faces and compute encodings for the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Only proceed if faces are detected
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        else:
            face_encodings = []
        
        # Initialize lists to store face names and matches
        face_names = []
        alert = False
        
        for face_encoding in face_encodings:
            # Compare detected face with known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                alert = True  # Set alert if the face is not recognized
            
            face_names.append(name)
        
        # Display results on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a rectangle around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Display the name of the person
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            # Show success message if recognized
            if name != "Unknown":
                cv2.putText(frame, "SUCCESS: Face Recognized", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Show alert if unknown face is detected
        if alert:
            cv2.putText(frame, "ALERT: Unrecognized Face Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)
        
        # Press 'q' to quit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Path to the folder containing images of authorized people
    known_faces_dir = "known_faces"  # Replace with your folder path
    
    # Load the known faces and their encodings
    known_encodings, known_names = load_known_faces(known_faces_dir)
    
    # Start real-time face detection and recognition
    recognize_faces(known_encodings, known_names)
