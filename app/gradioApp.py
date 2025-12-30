import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import numpy as np


# ================= 1. MODEL DEFINITIONS =================

class CustomBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x): return self.seq(x)


class ReconstructedCustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.block1 = CustomBlock(3, 32)
        self.block2 = CustomBlock(32, 64)
        self.block3 = CustomBlock(64, 128)
        self.block4 = CustomBlock(128, 256)
        self.block5 = CustomBlock(256, 512)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        return self.classifier(x)


def get_modified_resnet(num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(256, num_classes)
    )
    return model


# ================= 2. DYNAMIC CLASS LOADING =================

# Default fallback if nothing is found
CURRENT_CLASSES = ['Normal', 'Benign', 'Malignant (Stage 1)', 'Malignant (Stage 2)']


def load_classes_from_file(filename="classes.txt"):
    """Reloads the CURRENT_CLASSES list from the text file."""
    global CURRENT_CLASSES
    if os.path.exists(filename):
        print(f"✅ Found {filename}, reloading classes...")
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) > 0:
            CURRENT_CLASSES = lines
            return f"Loaded {len(lines)} classes: {', '.join(lines[:5])}..."
    return "No custom classes.txt found. Using defaults."


# Initial load on startup
load_classes_from_file()


def scan_and_update_classes(folder_path):
    """Scans a folder for subdirectories and updates classes.txt"""
    if not os.path.exists(folder_path):
        return "❌ Error: Folder path does not exist."

    if not os.path.isdir(folder_path):
        return "❌ Error: Path is not a directory."

    # Get all subfolders (standard PyTorch ImageFolder logic)
    # Filter out hidden files or non-directories
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Sort alphabetically (Critical! PyTorch does this)
    subdirs.sort()

    if not subdirs:
        return "⚠️ Error: No subfolders found in that directory."

    # Write to file
    with open("classes.txt", "w") as f:
        f.write('\n'.join(subdirs))

    # Update memory
    status = load_classes_from_file("classes.txt")

    return f"✅ Success! Scanned {len(subdirs)} classes.\nUpdated List: {', '.join(subdirs)}\n(Saved to classes.txt)"


# ================= 3. MODEL LOADING LOGIC =================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_models = {}


def load_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]

    path = model_name
    print(f"Loading {path}...")

    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Use the CURRENT globally loaded classes to determine architecture size
        n_classes = len(CURRENT_CLASSES)
        print(f"   -> Initializing model with {n_classes} output nodes.")

        # Auto-detect architecture
        if any('block1' in k for k in state_dict.keys()):
            model = ReconstructedCustomCNN(num_classes=n_classes)
        elif any('fc.4' in k for k in state_dict.keys()):
            model = get_modified_resnet(num_classes=n_classes)
        else:
            model = models.resnet18(weights=None)
            if model.fc.out_features != n_classes:
                model.fc = nn.Linear(model.fc.in_features, n_classes)

        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()

        loaded_models[model_name] = model
        return model
    except Exception as e:
        return f"Error: {e}"


# ================= 4. PREDICTION FUNCTION =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image, model_name, click_coords):
    if image is None:
        return "Please upload an image.", 0.0, None, None

    model = load_model(model_name)
    if isinstance(model, str): return model, 0.0, None, None

    h, w, _ = image.shape

    # 1. Determine Crop Center
    if click_coords:
        cx, cy = click_coords
    else:
        cy, cx = h // 2, w // 2

    # 2. Extract 64x64 Crop
    half_size = 32
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)

    image_crop = image[y1:y2, x1:x2]

    # 3. Visualization
    vis_image = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(vis_image)
    draw.rectangle([x1, y1, x2, y2], outline="yellow", width=5)

    # 4. Inference
    pil_crop = Image.fromarray(image_crop).convert('RGB')
    input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 5. Map to class name
    class_idx = predicted_idx.item()
    if class_idx < len(CURRENT_CLASSES):
        result_text = f"Prediction: {CURRENT_CLASSES[class_idx]}"
    else:
        result_text = f"Prediction: Class {class_idx} (Unknown Label)"

    return result_text, float(confidence.item()), image_crop, np.array(vis_image)


# ================= 5. UI SETUP =================

model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
if not model_files: model_files = ["No models found!"]


def on_select(evt: gr.SelectData):
    return evt.index


with gr.Blocks(title="Medical Inference Tool") as app:
    gr.Markdown("# 🏥 Medical Image Inference Tool")
    gr.Markdown("1. Upload an image.\n2. **Click anywhere** on the image to analyze that specific 64x64 spot.")

    click_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Input Image")
            model_selector = gr.Dropdown(choices=model_files, value=model_files[0] if model_files else None,
                                         label="Select Model")
            clear_btn = gr.Button("Clear Selection")

            # --- NEW: Folder Scan Section ---
            with gr.Accordion("📂 Advanced: Auto-Detect Classes from Dataset", open=False):
                gr.Markdown(
                    "Paste the path to your dataset folder (e.g., `/scratch1/.../patches_normalizedMC`) to automatically scan folder names.")
                folder_path_input = gr.Textbox(label="Dataset Folder Path", placeholder="/path/to/dataset")
                scan_btn = gr.Button("Scan & Update Classes")
                scan_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            result_label = gr.Label(label="Prediction Result")
            conf_bar = gr.Number(label="Confidence Score")
            with gr.Row():
                crop_view = gr.Image(label="Analyzed Patch (64x64)", interactive=False)
                full_view = gr.Image(label="Context View", interactive=False)

    # Prediction Logic
    image_input.select(on_select, None, click_state).then(
        fn=predict,
        inputs=[image_input, model_selector, click_state],
        outputs=[result_label, conf_bar, crop_view, full_view]
    )
    image_input.change(lambda: None, None, click_state)
    clear_btn.click(lambda: None, None, click_state)

    # Scan Logic
    scan_btn.click(
        fn=scan_and_update_classes,
        inputs=folder_path_input,
        outputs=scan_status
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)