import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the YOLO detection model
model_path = '..\\BTL Vision\\best.pt'
detection_model = YOLO(model_path)

# Load the classification model
classification_model = load_model('..\\BTL Vision\candefect_model.h5')

# Define classification classes
classes = ['Critical Defect', 'Major Defect', 'Minor Defect', 'No defect']

# Load the input image
image_path = '..\\BTL Vision\\input\\2.jpg'
image = cv2.imread(image_path)

# Resize the image (reduce size by 10%)
scale_percent = 10  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Perform object detection
results = detection_model(resized_image)

# Process and visualize results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes_detected = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    
    # Draw bounding boxes and classify each detected object
    for box, cls, conf in zip(boxes, classes_detected, confidences):
        x1, y1, x2, y2 = box
        
        # Extract the detected object
        detected_object = resized_image[y1:y2, x1:x2]
        
        # Prepare the object for classification
        if detected_object.size > 0:
            # Resize to match classification model input (299x299)
            resized_object = cv2.resize(detected_object, (299, 299))
            processed_object = img_to_array(resized_object)
            processed_object = np.expand_dims(processed_object, axis=0)
            processed_object /= 255.0  # Normalize pixel values
            
            # Classify the object
            prediction = classification_model.predict(processed_object)
            class_index = np.argmax(prediction)
            defect_class = classes[class_index]
            confidence_score = prediction[0][class_index]
            
            # Create label with detection and classification
            label = f'{detection_model.names[cls]} - {defect_class} {confidence_score:.2f}'
            
            # Draw rectangle for detection
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put label
            cv2.putText(resized_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display
output_image_path = '..\\BTL Vision\\output5.jpg'
cv2.imwrite(output_image_path, resized_image)
cv2.imshow('Detected and Classified Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()