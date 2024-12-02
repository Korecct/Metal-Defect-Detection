
# Metal Defect Detection

This repository provides a comprehensive framework for detecting surface defects using the NEU Surface Defect Database. The project encompasses dataset preparation, model training, evaluation, and real-time defect detection.

## Features

- **Dataset Preparation**: Download and organize the NEU Surface Defect Database.
- **Model Training**: Fine-tune a pre-trained ResNet18 model to classify defects.
- **Model Evaluation**: Evaluate model performance with metrics and confusion matrix visualization.
- **Batch and Real-Time Detection**: Perform defect detection on a batch of images or in real-time.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- Kaggle API
- NumPy
- Matplotlib
- Seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Dataset Preparation

Download and organize the NEU Surface Defect dataset:

```bash
python dataset_prep.py
```

### 2. Model Training

Train the defect detection model:

```bash
python model_train.py
```

### 3. Model Evaluation

Evaluate the trained model:

```bash
python model_evaluate.py
```

### 4. Real-Time and Batch Detection

Follow the prompts to use either batch or real-time detection.


```bash
python model_test.py
```

### 5. Configuration

Update paths and parameters in the respective scripts for dataset location, model checkpoint paths, and other configurations.

## Directory Structure

```
├── dataset_prep.py      # Dataset preparation script
├── model_train.py       # Model training script
├── model_evaluate.py    # Model evaluation script
├── model_test.py        # Real-time and batch detection script
├── utils.py             # Utility functions
├── data/                # Dataset storage
├── input_images/        # Input images for batch processing
├── outputs/             # Output results
└── requirements.txt     # Dependencies
```

## Results

### Confusion Matrix

The confusion matrix is saved as an image in the `outputs/` directory.

### Classification Report

The detailed classification report is stored as a text file in the `outputs/` directory.
