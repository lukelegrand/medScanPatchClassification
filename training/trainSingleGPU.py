import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import csv

# --- CONFIGURATION ---
DATA_DIR = r"/scratch1/llegrand/450final/patches_normalizedMC"
BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 0.001
LOG_DIR = './runs/single_gpu_experimentDeep2'
PERF_LOG_FILE = 'speed_metrics_single_gpuDeeper2.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. DATASET & DATALOADERS ---
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
class_names = full_dataset.classes

# Split (80/10/10)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- 2. MODEL DEFINITION ---
class VeryDeepMedicalCNN(nn.Module):
    def __init__(self, num_classes):
        super(VeryDeepMedicalCNN, self).__init__()
        
        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 5: 4x4 -> 2x2
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            # Input size: 512 filters * 2 * 2 spatial size = 2048
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Initialize the NEW Very Deep model
model = VeryDeepMedicalCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. TENSORBOARD & LOGGING INIT ---
writer = SummaryWriter(LOG_DIR)

# Initialize CSV for Speed Metrics
with open(PERF_LOG_FILE, 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['Epoch', 'Batch', 'Data_Load_Time', 'Forward_Time', 'Backward_Time', 'Optimizer_Time', 'Total_Batch_Time', 'Images_Per_Sec'])

# Add Images to TensorBoard
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = utils.make_grid(images[:16], normalize=True, nrow=4)
writer.add_image('Dataset_Samples', img_grid, 0)

# Add Graph to TensorBoard
dummy_input = torch.zeros(1, 3, 64, 64).to(device)
writer.add_graph(model, dummy_input)

# --- 4. TRAINING LOOP ---
print(f"Training Started. Logging metrics to {PERF_LOG_FILE}...")
start_time = time.time()  # Defined here as 'start_time' to match your expectation

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    t_end_prev = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        # 1. Data Load
        t_data_loaded = time.time()
        data_load_time = t_data_loaded - t_end_prev

        inputs, targets = inputs.to(device), targets.to(device)

        # 2. Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        t_forward_done = time.time()
        forward_time = t_forward_done - t_data_loaded

        # 3. Backward
        loss.backward()
        t_backward_done = time.time()
        backward_time = t_backward_done - t_forward_done

        # 4. Optimizer
        optimizer.step()
        torch.cuda.synchronize() 
        t_opt_done = time.time()
        opt_time = t_opt_done - t_backward_done

        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # CSV Log
        total_batch_time = time.time() - t_end_prev
        throughput = BATCH_SIZE / total_batch_time
        
        with open(PERF_LOG_FILE, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, i, f"{data_load_time:.4f}", f"{forward_time:.4f}", f"{backward_time:.4f}", f"{opt_time:.4f}", f"{total_batch_time:.4f}", f"{throughput:.2f}"])

        t_end_prev = time.time()

    # Epoch Metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    # Validation
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
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# --- 5. EVALUATION ---
total_time = time.time() - start_time
print(f"Training Finished in {total_time/60:.2f} minutes")
writer.close()

model.eval()

# ---------------------------------------------------------
# PART A: 100-Class Confusion Matrix
# ---------------------------------------------------------
print("Generating 100-Class Confusion Matrix...")

# Combine the two datasets
fair_dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
full_loader = DataLoader(fair_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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

# Calculate dynamic size: 100 classes needs a BIG plot
plt.figure(figsize=(40, 40)) # Huge size for 100 classes
sns.heatmap(
    cm, 
    annot=True,         # Show numbers
    fmt='d',            # Integers
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names,
    annot_kws={"size": 8},    # Small font for numbers inside cells
    cbar=False                # Hide color bar to save space
)

plt.xticks(rotation=90, fontsize=8) # Rotate x labels
plt.yticks(fontsize=8)              # Small y labels
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.title('Confusion Matrix (100 Classes)', fontsize=16)
plt.tight_layout()
plt.savefig('confusion_matrix_100_classesDeep2.png', dpi=150) # High DPI for zooming in
print("Large Confusion Matrix saved to 'confusion_matrix_100_classes2.png'")


# ---------------------------------------------------------
# PART B: 10x10 Test Image Grid (100 Images)
# ---------------------------------------------------------
print("Generating 10x10 Prediction Grid...")

# Get 100 samples from the test dataset
# We use the dataset directly to get individual items easily
indices = np.random.choice(len(test_dataset), 100, replace=False)
samples = [test_dataset[i] for i in indices]

# Prepare plot
fig, axes = plt.subplots(10, 10, figsize=(20, 20)) # 20x20 inch plot
axes = axes.flatten()

for i, (image, label) in enumerate(samples):
    # 1. Run Prediction
    img_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        pred_idx = pred.item()

    # 2. Unnormalize image for display
    # (image was norm to mean=0.5, std=0.5 -> unnorm is img*0.5 + 0.5)
    img_disp = image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    img_disp = np.clip(img_disp, 0, 1) # Ensure valid range

    # 3. Plot
    ax = axes[i]
    ax.imshow(img_disp)
    
    # Color code title: Green if correct, Red if wrong
    true_name = class_names[label]
    pred_name = class_names[pred_idx]
    color = 'green' if label == pred_idx else 'red'
    
    # Shorten names for display if too long
    if len(pred_name) > 10: pred_name = pred_name[:8] + ".."
    
    ax.set_title(f"{pred_name}", color=color, fontsize=8)
    ax.axis('off')

# Very small margins as requested
plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.01, right=0.99, bottom=0.01, top=0.95)
plt.suptitle("Test Predictions: Green = Correct, Red = Wrong", fontsize=16)
plt.savefig('test_predictions_gridDeeper2.png')
print("10x10 Grid saved to 'test_predictions_gridDD.png'")

# Save Final Model
torch.save(model.state_dict(), 'model_single_gpu2.pth')