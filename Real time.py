import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the YOLO detection model
model_path = '..\\BTL Vision\\best.pt'
detection_model = YOLO(model_path)

# Load the defect classification model
classification_model = load_model('..\\BTL Vision\\candefect_model.h5')

# Define classification classes
classes = ['Critical Defect', 'Major Defect', 'Minor Defect', 'No Defect']

# Start the webcam (or any connected camera)
video_capture = cv2.VideoCapture(0)  # Use default camera (0 is the camera ID)

if not video_capture.isOpened():
    print("Unable to open the camera")
    exit()

# Process the video stream
while True:
    ret, frame = video_capture.read()  # Read each frame
    if not ret:
        print("Unable to read frames from the camera")
        break

    # Resize the frame to improve processing speed
    scale_percent = 50  # Reduce frame size by 50%
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Perform object detection on the frame
    results = detection_model(resized_frame)

    # Process each detected object
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Bounding box coordinates
        classes_detected = result.boxes.cls.cpu().numpy().astype(int)  # Detected classes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        for box, cls, conf in zip(boxes, classes_detected, confidences):
            x1, y1, x2, y2 = box
            
            # Extract the detected object from the frame
            detected_object = resized_frame[y1:y2, x1:x2]

            # Classify the object if it has a valid size
            if detected_object.size > 0:
                # Resize the object to 299x299 to fit the classification model
                resized_object = cv2.resize(detected_object, (299, 299))
                processed_object = img_to_array(resized_object)
                processed_object = np.expand_dims(processed_object, axis=0)
                processed_object = processed_object / 255.0  # Normalize pixel values

                # Classify the object
                prediction = classification_model.predict(processed_object)
                class_index = np.argmax(prediction)  # Class with the highest probability
                defect_class = classes[class_index]  # Class name
                confidence_score = prediction[0][class_index]  # Probability of the class

                # Create a label with detection and classification results
                label = f'{detection_model.names[cls]} - {defect_class} {confidence_score:.2f}'
            else:
                # If classification fails, show detection results only
                label = f'{detection_model.names[cls]} {conf:.2f}'
            
            # Draw the bounding box
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add the label to the frame
            cv2.putText(resized_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Real-Time Detection and Classification', resized_frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()