import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import csv
import copy

# --- CONFIGURATION ---
DATA_DIR = r"/scratch1/llegrand/450final/patches_normalizedMC"
BATCH_SIZE = 64
NUM_EPOCHS = 100         # Reduced, Early Stopping will catch it earlier
LEARNING_RATE = 0.0005   # Lower LR because we are fine-tuning
WEIGHT_DECAY = 1e-3      # STRONGER Regularization (was 1e-4)
LOG_DIR = './runs/resnet_frozen'
PERF_LOG_FILE = 'speed_metrics_frozen.csv'
PATIENCE = 15            # Stop if no improvement for 15 epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. DATASET & DATALOADERS ---

# STRONGER Augmentation: RandAugment
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=9), # <--- NEW: Heavy Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None) 
class_names = full_dataset.classes
num_classes = len(class_names)

# Split (80/10/10)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

# Wrapper to apply specific transforms
class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

train_dataset = SubsetWrapper(train_subset, transform=train_transforms)
val_dataset   = SubsetWrapper(val_subset, transform=val_test_transforms)
test_dataset  = SubsetWrapper(test_subset, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- 2. MODEL DEFINITION (FROZEN BACKBONE) ---
print("Initializing ResNet18 (Frozen Backbone)...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# --- NEW: FREEZE ALL LAYERS FIRST ---
for param in model.parameters():
    param.requires_grad = False

# Replace head (These new layers are trainable by default)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),   # Increased width
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.6),            # Higher Dropout (was 0.5)
    nn.Linear(512, num_classes)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Only optimize parameters that require gradients (the head)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

# --- 3. TRAINING LOOP ---
writer = SummaryWriter(LOG_DIR)
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # --- VALIDATE ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
    
    val_acc = 100 * val_correct / val_total
    
    # Logging
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    scheduler.step(val_acc)

    # --- SAVE BEST & EARLY STOPPING ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model_frozen.pth')
        print(f"--> New Best Model Saved! (Val Acc: {val_acc:.2f}%)")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs.")

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered.")
        break

# Finish
print(f"Training Finished. Best Validation Acc: {best_val_acc:.2f}%")
writer.close()

# Load best weights for final evaluation
model.load_state_dict(best_model_wts)

# ---------------------------------------------------------
# PART A: Confusion Matrix
# ---------------------------------------------------------
print(f"Generating Confusion Matrix for {num_classes} classes...")

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(40, 40))
sns.heatmap(
    cm, 
    annot=True,         
    fmt='d',            
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names,
    annot_kws={"size": 8},    
    cbar=False                
)

plt.xticks(rotation=90, fontsize=8) 
plt.yticks(fontsize=8)              
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.title(f'Confusion Matrix (Frozen Backbone)', fontsize=16)
plt.tight_layout()
plt.savefig('confusion_matrix_frozen.png', dpi=150)
print("Large Confusion Matrix saved to 'confusion_matrix_frozen.png'")


# ---------------------------------------------------------
# PART B: 10x10 Test Image Grid
# ---------------------------------------------------------
print("Generating 10x10 Prediction Grid...")

indices = np.random.choice(len(test_dataset), 100, replace=False)
samples = [test_dataset[i] for i in indices]

fig, axes = plt.subplots(10, 10, figsize=(20, 20)) 
axes = axes.flatten()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, (image, label) in enumerate(samples):
    img_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        pred_idx = pred.item()

    img_disp = image.permute(1, 2, 0).cpu().numpy()
    img_disp = std * img_disp + mean
    img_disp = np.clip(img_disp, 0, 1) 

    ax = axes[i]
    ax.imshow(img_disp)
    
    true_name = class_names[label]
    pred_name = class_names[pred_idx]
    color = 'green' if label == pred_idx else 'red'
    
    if len(pred_name) > 10: pred_name = pred_name[:8] + ".."
    
    ax.set_title(f"{pred_name}", color=color, fontsize=8)
    ax.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, bottom=0.01, top=0.95)
plt.suptitle("Frozen ResNet Test Predictions", fontsize=16)
plt.savefig('test_predictions_grid_frozen.png')
print("10x10 Grid saved to 'test_predictions_grid_frozen.png'")

torch.save(model.state_dict(), 'model_resnet_frozen.pth')