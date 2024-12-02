import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys
import subprocess
import cv2
import numpy as np
from dataset_prep import download_dataset
from model_train import get_model
from utils import load_data, evaluate_model, detect_defect_in_image, real_time_detection
from PIL import Image
from datetime import datetime

def batch_process_images(model, class_names, device, input_dir='input_images', output_dir='output', threshold=0.8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files in input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No images found in {input_dir}.")
    else:
        print(f"Processing {len(image_files)} images from {input_dir}.")

        model.eval()

        for image_name in image_files:
            image_path = os.path.join(input_dir, image_name)
            img = Image.open(image_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(img_t)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                # Get top predicted class and its probability
                top_prob, top_class = torch.max(probabilities, dim=0)
                predicted_class = class_names[top_class]

            # Convert probability to percentage
            confidence_percentage = top_prob.item() * 100

            # Get current date and time
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Determine PASS or FAIL based on threshold
            if top_prob >= threshold:
                result = f'Defect Detected: {predicted_class} (FAIL)'
                text_color = (0, 0, 255)  # Red color for FAIL
            else:
                result = f'No Defect Detected (PASS)'
                text_color = (0, 255, 0)  # Green color for PASS

            annotation_text = (
                f"Result: {result}",
                f"Confidence: {confidence_percentage:.2f}%",
                f"Processed On: {current_datetime}"
            )

            # Convert the image to OpenCV format for annotation
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Scale annotations based on image size
            img_height, img_width = img_cv.shape[:2]
            font_scale = max(img_height, img_width) / 1000.0 
            thickness = max(int(max(img_height, img_width) / 500), 1)

            # Annotate the image
            y0, dy = int(30 * font_scale), int(40 * font_scale)
            for i, line in enumerate(annotation_text):
                y = y0 + i * dy
                cv2.putText(img_cv, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, text_color, thickness, cv2.LINE_AA)

            # Rename the output image to include classification type and confidence percentage

            confidence_str = f"{int(confidence_percentage)}"

            # Construct the new image name
            new_image_name = f"{confidence_str}%_{predicted_class}.jpg"

            output_image_path = os.path.join(output_dir, new_image_name)

            # Save the annotated image to output directory
            cv2.imwrite(output_image_path, img_cv)

            print(f"Processed {image_name}, saved to {output_image_path}")

def main():
    # Define dataset identifier and paths
    neu_dataset = 'kaustubhdikshit/neu-surface-defect-database'
    download_path = 'data'
    model_save_path = 'defect_detection_model.pth'
    output_dir = 'test_outputs'

    # Check if dataset is already prepared
    dataset_prepared = os.path.isdir('data/dataset')
    
    if not dataset_prepared:
        # Download NEU dataset if not already downloaded
        try:
            if not os.path.isdir(os.path.join(download_path, 'NEU-DET')):
                download_dataset(neu_dataset, download_path)
        except Exception as e:
            print(f"Error downloading NEU dataset: {e}")
            sys.exit(1)

        # Define extracted dataset path
        neu_dataset_path = os.path.join(download_path, 'NEU-DET')
        if not os.path.isdir(neu_dataset_path):
            # Attempt to find the extracted folder
            extracted_folders = [f for f in os.listdir(download_path) if os.path.isdir(os.path.join(download_path, f))]
            if extracted_folders:
                neu_dataset_path = os.path.join(download_path, extracted_folders[0])
            else:
                print(f"Error: Extracted NEU dataset directory not found in {download_path}.")
                sys.exit(1)

        # Prepare the dataset
        dataset_preparation_script = 'dataset_prep.py'
        if not os.path.exists(dataset_preparation_script):
            print(f'Error: {dataset_preparation_script} not found.')
            sys.exit(1)
        
        try:
            subprocess.run([sys.executable, dataset_preparation_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error running {dataset_preparation_script}: {e}')
            sys.exit(1)
        
        print("Dataset prepared and split into train, val, and test sets.")
    else:
        print("Dataset already downloaded and prepared. Skipping download and preparation.")

    # Set data directory
    data_dir = 'data/dataset'

    # Load data
    try:
        dataloaders, dataset_sizes, class_names = load_data(data_dir)
    except Exception as e:
        print(f'Error loading data: {e}')
        sys.exit(1)

    if not class_names:
        print('No classes found in the dataset. Exiting.')
        sys.exit(1)

    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize the model
    model = get_model(num_classes)
    model = model.to(device)

    try:
        model_state = torch.load(model_save_path, map_location=device, weights_only=True)
    except TypeError:
        model_state = torch.load(model_save_path, map_location=device)
        print("Warning: 'weights_only' parameter not supported in this PyTorch version.")

    # Load the trained model weights
    try:
        model.load_state_dict(model_state)
        print(f"Loaded trained model from {model_save_path}")
    except RuntimeError as e:
        print(f"Error loading the trained model: {e}")
        print("Attempting to load with strict=False")
        try:
            model.load_state_dict(model_state, strict=False)
            print("Model loaded with missing/unexpected keys ignored.")
        except Exception as e2:
            print(f"Failed to load model with strict=False: {e2}")
            sys.exit(1)

    # Ask the user whether to perform batch processing or real-time detection
    while True:
        user_input = input("Enter 'batch' to process images in the input_images folder, 'realtime' for real-time detection, or 'exit' to quit: ").strip().lower()
        if user_input == 'batch':
            # Ask user for threshold value
            while True:
                threshold_input = input("Enter the confidence threshold (e.g., 0.8): ").strip()
                try:
                    threshold = float(threshold_input)
                    if 0.0 <= threshold <= 1.0:
                        break
                    else:
                        print("Please enter a value between 0 and 1.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value between 0 and 1.")

            # Batch process images in 'input_images' folder
            input_images_dir = 'input_images'
            output_images_dir = 'output'
            batch_process_images(model, class_names, device, input_dir=input_images_dir, output_dir=output_images_dir, threshold=threshold)
            break
        elif user_input == 'realtime':
            # Ask user for threshold value
            while True:
                threshold_input = input("Enter the confidence threshold (e.g., 0.8): ").strip()
                try:
                    threshold = float(threshold_input)
                    if 0.0 <= threshold <= 1.0:
                        break
                    else:
                        print("Please enter a value between 0 and 1.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value between 0 and 1.")

            # Real-time detection with user-specified threshold
            real_time_detection(model, class_names, device, camera_id=0, threshold=threshold)
            break
        elif user_input == 'exit':
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid input. Please enter 'batch', 'realtime', or 'exit'.")

if __name__ == '__main__':
    main()
