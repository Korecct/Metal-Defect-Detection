import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import zipfile
import kaggle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def download_and_extract_dataset(dataset='kaustubhdikshit/neu-surface-defect-database', download_path='data'):
    os.makedirs(download_path, exist_ok=True)
    print(f'Downloading dataset: {dataset}')
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=False)

    # Identify the downloaded zip file
    zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"ZIP file for dataset {dataset} not found in {download_path}.")
    zip_file = os.path.join(download_path, zip_files[0])

    print(f'Extracting {zip_file}...')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_file)
    print(f'Dataset {dataset} downloaded and extracted.')

def load_data(data_dir, batch_size=32, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Check if data_dir exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The data directory '{data_dir}' does not exist. Please run 'dataset_prep.py' first.")

    # Load datasets
    image_datasets = {}
    available_splits = ['train', 'val']
    available_classes = set()

    for split in available_splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f'Warning: Split directory {split_dir} does not exist. Skipping...')
            continue
        image_datasets[split] = datasets.ImageFolder(
            split_dir, 
            data_transforms[split]
        )
        available_classes.update(image_datasets[split].classes)

    # Create data loaders
    dataloaders = {}
    dataset_sizes = {}
    class_names = []

    for split in available_splits:
        if split in image_datasets:
            shuffle = True if split == 'train' else False
            dataloaders[split] = DataLoader(
                image_datasets[split], 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers
            )
            dataset_sizes[split] = len(image_datasets[split])
            if split == 'train':
                class_names = image_datasets[split].classes

    if not dataloaders:
        raise ValueError("No data loaders were created. Please check the dataset preparation.")

    return dataloaders, dataset_sizes, class_names

def evaluate_model(model, dataloaders, class_names, device, output_dir='outputs'):
    if 'val' not in dataloaders:
        print('Validation data loader not found. Skipping evaluation.')
        return

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    classification_rep = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(classification_rep)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save Classification Report as Text File
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_rep)
    print('Classification report saved to outputs/classification_report.txt')

    # Confusion Matrix and Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print('Confusion matrix saved to outputs/confusion_matrix.png')

def detect_defect_in_image(model, image_path, class_names, device):

    # Define transformation for image inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]

    # Determine PASS or FAIL
    result = f'Defect Detected: {predicted_class} (FAIL)'

    # Output result
    print(result)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(result)
    plt.show()

def real_time_detection(model, class_names, device, camera_id=0, threshold=0.8):
    # Define transformation for real-time inference
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    model.eval()
    print("Starting real-time defect detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame
        img_t = transform(frame_rgb).unsqueeze(0).to(device)

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
            result = 'No Defect Detected (PASS)'
            text_color = (0, 255, 0)  # Green color for PASS

        annotation_text = (
            f"Result: {result}",
            f"Confidence: {confidence_percentage:.2f}%",
            f"Time: {current_datetime}"
        )

        # Annotate the frame
        y0, dy = 30, 30  # Starting y position and line height
        for i, line in enumerate(annotation_text):
            y = y0 + i * dy
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, text_color, 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Real-Time Defect Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting real-time detection.")
            break

    cap.release()
    cv2.destroyAllWindows()