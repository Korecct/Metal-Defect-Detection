import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import sys
from dataset_prep import download_dataset
from model_train import get_model, train_model
from utils import load_data, evaluate_model, detect_defect_in_image, real_time_detection
from torch.utils.tensorboard import SummaryWriter
import subprocess
from tqdm import tqdm

def main():
    neu_dataset = 'kaustubhdikshit/neu-surface-defect-database'

    download_path = 'data'

    # Download NEU dataset
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

    # Get model
    model = get_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/defect_detection_experiment')

    # Train the model with checkpointing and early stopping
    num_epochs = 25
    checkpoint_path = 'model_checkpoint.pth'  # Define a checkpoint path
    model, history = train_model(
        model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device,
        num_epochs=num_epochs, writer=writer, checkpoint_path=checkpoint_path, early_stopping_patience=5
    )

    # Save the trained model
    model_save_path = 'defect_detection_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved as {model_save_path}')

    # Close the TensorBoard writer
    writer.close()

    # Evaluate the model
    evaluate_model(model, dataloaders, class_names, device, output_dir='outputs')

    # Real-time detection
    run_realtime = False
    if run_realtime:
        real_time_detection(model, class_names, device)

    # Image detection
    run_image_detection = False
    if run_image_detection:
        image_path = 'image.jpg'
        detect_defect_in_image(model, image_path, class_names, device)

if __name__ == '__main__':
    main()