import os
import shutil
import sys
import zipfile
import kaggle

def download_dataset(dataset, download_path):
    print(f'Downloading dataset: {dataset}')
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=False)

    # Identify the downloaded zip file
    zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No ZIP file found for dataset {dataset} in {download_path}.")
    zip_file = os.path.join(download_path, zip_files[0])

    print(f'Extracting {zip_file}...')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_file)
    print(f'Dataset {dataset} downloaded and extracted.')

def prepare_dataset(neu_dataset_path, output_dir='data/dataset'):
    
    split_dirs = {
        'train': 'train',
        'val': 'validation'
    }

    # Define the defect classes 
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    splits = ['train', 'val']

    # Clean the output directory
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create output directories
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # Iterate over each split directory
    for split in splits:
        split_dir_name = split_dirs[split]
        images_dir = os.path.join(neu_dataset_path, split_dir_name, 'images')

        if not os.path.isdir(images_dir):
            print(f'Warning: Images directory {images_dir} does not exist. Skipping split "{split}".')
            continue

        print(f'Processing split: {split}')

        for class_name in classes:
            class_dir = os.path.join(images_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f'Warning: Directory {class_dir} does not exist. Skipping class "{class_name}".')
                continue

            class_images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not class_images:
                print(f'Warning: No images found in {class_dir}. Skipping class "{class_name}".')
                continue

            # Copy images to the organized dataset directory
            for img in class_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(output_dir, split, class_name, img)
                shutil.copyfile(src, dst)

    print('Dataset prepared and split into train and val sets.')

if __name__ == '__main__':
    # Kaggle dataset 
    neu_dataset = 'kaustubhdikshit/neu-surface-defect-database'

    download_path = 'data'
    
    try:
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
    prepare_dataset(neu_dataset_path)