import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm

# Paths and constants
test_csv = r'C:\Users\ADMIN\Downloads\BTL Vision\canned-food-surface-defect.v6-last-version.yolov11\test\_classes.csv'

test_dir = r'C:\Users\ADMIN\Downloads\BTL Vision\canned-food-surface-defect.v6-last-version.yolov11\test\images'

model_path = r'..\BTL Vision\candefect_model.h5'

yolo_model_path = r'..\BTL Vision\best.pt'

img_height, img_width = 299, 299
y_columns = ['Critical Defect', 'Major Defect', 'Minor Defect', 'No defect']

# Load the CSV file
test_df = pd.read_csv(test_csv)
test_df.columns = test_df.columns.str.strip()

# Load trained Xception model
model = load_model(model_path)

# Load YOLO model for object detection
yolo_model = YOLO(yolo_model_path)

# Data generator for test data (apply the same rescale factor as used during training)
test_datagen = ImageDataGenerator(rescale=1./255)

# Function to crop image using YOLO
def crop_image_using_yolo(image_path):
    """
    Detect and crop can from image using YOLOv8.
    Resizes the cropped region to the desired input size for the classifier.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} not found.")
        return None, None
    
    # Run YOLO prediction to find objects in the image
    results = yolo_model(image)
    
    if len(results[0].boxes) > 0:
        # Take the first detected box
        box = results[0].boxes[0]
        confidence = float(box.conf)
        
        if confidence > 0.5:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Crop the image using bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size > 0:
                # Resize cropped image to match model input size
                cropped_image = cv2.resize(cropped_image, (img_width, img_height))
                return cropped_image, (x1, y1, x2, y2)
    
    # If no valid detection, return None
    print(f"No valid detection for {image_path}")
    return None, None

# Predict and draw bounding boxes
def predict_and_draw_bboxes(test_df, source_dir, model):
    """
    Predict defects using the Xception model and draw bounding boxes.
    """
    true_labels = []
    predicted_labels = []
    incorrect_indices = []
    correct_indices = []

    with tqdm(total=len(test_df), desc="Processing test images") as pbar:
        for idx, row in test_df.iterrows():
            image_filename = row['filename']
            true_label = row[y_columns].values  # Ground truth labels
            image_path = os.path.join(source_dir, image_filename)

            # Crop the image using YOLO
            cropped_image, bbox_coords = crop_image_using_yolo(image_path)

            if cropped_image is not None:
                # Rescale the image after resizing (value range 0-1)
                cropped_image = cropped_image / 255.0
                cropped_image = np.expand_dims(cropped_image, axis=0)

                # Predict using the trained Xception model
                prediction = model.predict(cropped_image)
                predicted_label = np.argmax(prediction, axis=1)[0]
                true_label_idx = np.argmax(true_label)

                # Record true and predicted labels
                true_labels.append(true_label_idx)
                predicted_labels.append(predicted_label)

                # Determine correct or incorrect prediction
                if true_label_idx == predicted_label:
                    correct_indices.append(idx)
                else:
                    incorrect_indices.append(idx)

            pbar.update(1)

    return true_labels, predicted_labels, correct_indices, incorrect_indices

# Process test images and make predictions
print("Processing test images and making predictions...")
true_labels, predicted_labels, correct_indices, incorrect_indices = predict_and_draw_bboxes(test_df, test_dir, model)

# Calculate accuracy and other metrics
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy on the test set: {accuracy:.4f}")

# Generate a classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=y_columns))
# Function to plot images
def plot_images(indices, title, test_df, directory):
    plt.figure(figsize=(16, 16))
    for i, idx in enumerate(indices[:9]):
        filename = test_df.iloc[idx]['filename']
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        if image is not None:
            plt.subplot(3, 3, i + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"{title}\n{filename}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot incorrect predictions
print("\nPlotting incorrect predictions...")
plot_images(incorrect_indices, "Incorrect Predictions", test_df, test_dir)

# Plot correct predictions
print("\nPlotting correct predictions...")
plot_images(correct_indices, "Correct Predictions", test_df, test_dir)

# Tính Precision, Recall, F1-Score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

def plot_metrics(precision, recall, f1, class_names):
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width=width, label='Precision', color='skyblue')
    plt.bar(x, recall, width=width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width=width, label='F1-Score', color='salmon')

    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Precision, Recall và F1-Score cho từng lớp', fontsize=16)
    plt.xticks(x, class_names, fontsize=12, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ Precision, Recall, F1
plot_metrics(precision, recall, f1, y_columns)

# Plot comparison of correct vs incorrect predictions
def plot_correct_vs_incorrect(correct_count, incorrect_count):
    """
    Plot a bar chart comparing correct and incorrect predictions.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Correct', 'Incorrect'], y=[correct_count, incorrect_count], palette='viridis')
    plt.title("Comparison of Correct vs Incorrect Predictions", fontsize=16)
    plt.ylabel("Count", fontsize=14)
    plt.xlabel("Prediction Type", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Tính toán số lượng correct và incorrect
correct_count = len(correct_indices)
incorrect_count = len(incorrect_indices)

# Gọi hàm vẽ biểu đồ
plot_correct_vs_incorrect(correct_count, incorrect_count)
