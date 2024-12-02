import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import copy
from tqdm import tqdm
import os

def get_model(num_classes, dropout_p=0.5):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    num_ftrs = model.fc.in_features
    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, writer=None, checkpoint_path='checkpoint.pth', early_stopping_patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase not in dataloaders:
                print(f'Phase {phase} not found. Skipping...')
                continue

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Wrap the DataLoader with tqdm for progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase", unit="batch")

            # Iterate over data
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar description
                current_loss = running_loss / ((running_corrects.item() / preds.size(0)) if preds.size(0) > 0 else 1)
                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log to TensorBoard
            if writer:
                if phase == 'train':
                    writer.add_scalar('Loss/Train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
                else:
                    writer.add_scalar('Loss/Val', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/Val', epoch_acc, epoch)

            # Check for improvement
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    # Save the best checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                        'history': history
                    }, checkpoint_path)
                    print(f"Checkpoint saved at epoch {epoch+1}")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s)")

        print()

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f'Training complete. Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history