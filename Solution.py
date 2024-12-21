!pip install ultralytics

#Train: 

import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve, f1_score
from tqdm import tqdm

# Parameters
img_height, img_width = 299, 299
batch_size = 32
l2_lambda = 0.001  # L2 regularization strength

# Load the CSV files
train_csv = '/kaggle/input/candataset/train/_classes.csv'
valid_csv = '/kaggle/input/candataset/valid/_classes.csv'
train_df = pd.read_csv(train_csv)
valid_df = pd.read_csv(valid_csv)

# Remove extra whitespace from column names
train_df.columns = train_df.columns.str.strip()
valid_df.columns = valid_df.columns.str.strip()

# Columns for the labels
y_columns = ['Critical Defect', 'Major Defect', 'Minor Defect', 'No defect']

# Load YOLO model for cropping
yolo_model = YOLO('/kaggle/input/can-detection/pytorch/default/1/best.pt')

def crop_image_using_yolo(image_path):
    """
    Detect and crop can from image using YOLOv8, then convert to grayscale.
    Only processes images where a 'Body' is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} not found.")
        return None, None, None

    results = yolo_model(image)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        confidence = float(box.conf)

        # Get the class name
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        # Only process if it's a 'Body' with sufficient confidence
        if class_name == 'Body' and confidence > 0.5:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped_image = image[y1:y2, x1:x2]

            if cropped_image.size > 0:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                gray_3channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                resized_image = cv2.resize(gray_3channel, (img_width, img_height))
                return resized_image, (x1, y1, x2, y2), cropped_image
        else:
            print(f"Skipping {image_path}: Detected {class_name} instead of Body")
            return None, None, None

    print(f"No valid detection for {image_path}")
    return None, None, None

def process_and_crop_images(df, source_dir):
    """
    Process all images in the dataframe using YOLO detection silently.
    """
    updated_filenames = []
    cropped_count = 0
    total_count = len(df)

    for _, row in df.iterrows():
        image_filename = row['filename']
        image_path = os.path.join(source_dir, image_filename)

        cropped_image = crop_image_using_yolo(image_path)

        if cropped_image is not None:
            cv2.imwrite(image_path, cropped_image)
            cropped_count += 1

        updated_filenames.append(image_filename)

    df['filename'] = updated_filenames
    return df, cropped_count, total_count

# Process and crop the training and validation datasets
print("Processing training images...")
train_df, train_cropped_count, train_total_count = process_and_crop_images(train_df, '/kaggle/input/candataset/train')
print("\nProcessing validation images...")
valid_df, valid_cropped_count, valid_total_count = process_and_crop_images(valid_df, '/kaggle/input/candataset/valid')

# Print summary of cropping results
print(f"\nTraining set: Cropped {train_cropped_count} out of {train_total_count} images")
print(f"Validation set: Cropped {valid_cropped_count} out of {valid_total_count} images")

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Prepare generators for training and validation
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/input/candataset/train',
    x_col='filename',
    y_col=y_columns,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw'
)
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory='/kaggle/input/candataset/valid',
    x_col='filename',
    y_col=y_columns,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='raw'
)

# Build the Xception model for classification with regularization
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),  # Batch normalization for better training stability
    Dense(1024, activation='relu', kernel_regularizer=l2(l2_lambda)),  # L2 regularization applied here
    Dropout(0.5),  # Dropout for regularization, helps to prevent overfitting
    Dense(4, activation='softmax', kernel_regularizer=l2(l2_lambda))  # L2 regularization applied to output layer
])
base_model.trainable = False

# Class balancing
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_generator.y), y=train_generator.y)
class_weight_dict = dict(zip(np.unique(train_generator.y), class_weights))

# Compile the model with regularization in mind
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with class weights
print("\nTraining with frozen base layers...")
history = model.fit(
    train_generator,
    epochs=4,
    validation_data=valid_generator,
    class_weight=class_weight_dict,
    verbose=1
)

# Fine-tune the model
print("\nFine-tuning the model...")
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    epochs=3,
    validation_data=valid_generator,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate the model
y_pred = model.predict(valid_generator)
precision, recall, thresholds = precision_recall_curve(valid_generator.y, y_pred, pos_label=1)
f1_scores = [2 * p * r / (p + r + 1e-8) for p, r in zip(precision, recall)]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold:.2f}")

y_pred_binary = (y_pred[:, 1] > best_threshold).astype(int)
valid_f1 = f1_score(valid_generator.y, y_pred_binary)
print(f"Validation F1-score: {valid_f1:.4f}")

# Save the model and weights
model_save_path = '/kaggle/working/candefect_model.h5'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

weights_save_path = '/kaggle/working/candefect_weights.weights.h5'
model.save_weights(weights_save_path)
print(f"Weights saved at {weights_save_path}")

#Test: 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Paths and constants
test_csv = '/kaggle/input/candataset/test/_classes.csv'
test_dir = '/kaggle/input/candataset/test'
model_path = '/kaggle/working/candefect_model.h5'
yolo_model_path = '/kaggle/input/can-detection/pytorch/default/1/best.pt'

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
def predict_and_draw_bboxes(test_df, source_dir, model):
    """
    Predict defects using the model and draw bounding boxes on original images.
    Only processes and visualizes images where a 'Body' is detected.
    Skips all non-Body images from prediction.
    """
    true_labels = []
    predicted_labels = []
    incorrect_indices = []
    correct_indices = []
    skipped_count = 0
    processed_count = 0
    
    # Create a filtered dataframe to store only valid predictions
    filtered_df = pd.DataFrame(columns=test_df.columns)

    with tqdm(total=len(test_df), desc="Processing test images") as pbar:
        for idx, row in test_df.iterrows():
            image_filename = row['filename']
            true_label = row[y_columns].values
            image_path = os.path.join(source_dir, image_filename)

            # Get both grayscale and original versions
            cropped_gray, bbox_coords, cropped_original = crop_image_using_yolo(image_path)

            if cropped_gray is not None:
                processed_count += 1
                # Use grayscale version for prediction
                cropped_gray = cropped_gray / 255.0
                cropped_gray = np.expand_dims(cropped_gray, axis=0)

                prediction = model.predict(cropped_gray)
                predicted_label = np.argmax(prediction, axis=1)[0]
                true_label_idx = np.argmax(true_label)

                true_labels.append(true_label_idx)
                predicted_labels.append(predicted_label)

                # Add row to filtered dataframe
                filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)

                if true_label_idx == predicted_label:
                    correct_indices.append(idx)
                else:
                    incorrect_indices.append(idx)

                # Draw bounding box on the original image
                if bbox_coords is not None:
                    x1, y1, x2, y2 = bbox_coords
                    original_image = cv2.imread(image_path)
                    # Draw bbox with green for correct predictions, red for incorrect
                    color = (0, 255, 0) if true_label_idx == predicted_label else (0, 0, 255)
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label_text = f"Pred: {y_columns[predicted_label]}, True: {y_columns[true_label_idx]}"
                    cv2.putText(original_image, label_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Save image with bbox
                    os.makedirs("/kaggle/working/results", exist_ok=True)
                    output_path = f"/kaggle/working/results/{image_filename}"
                    cv2.imwrite(output_path, original_image)
            else:
                skipped_count += 1

            pbar.update(1)

    print(f"\nProcessing Summary:")
    print(f"Processed images: {processed_count}")
    print(f"Skipped images: {skipped_count}")
    print(f"Total images: {len(test_df)}")
    print(f"Images with Body class: {len(filtered_df)}")
    print(f"Correct predictions: {len(correct_indices)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}")

    return true_labels, predicted_labels, correct_indices, incorrect_indices, filtered_df
