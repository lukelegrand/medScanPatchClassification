import torch
import os
import glob
import time
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
# Path to your folder containing .pth or .pt files
MODEL_FOLDER = './models' 

# Define your Model Architecture here
# form your_model_file import YourModelClass
import torchvision.models as models

def get_model_architecture():
    # REPLACE this with your actual model initialization
    # Example: return MyCustomModel()
    return models.resnet18(pretrained=False) 

# Input shape for inference (Batch_Size, Channels, Height, Width)
INPUT_SHAPE = (1, 3, 224, 224) 

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =================================================

def measure_inference_time(model_path):
    # 1. Initialize Model
    model = get_model_architecture()
    
    # 2. Load Weights
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle cases where checkpoint is a dict (common in training)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
             # Try loading directly if keys match, otherwise might need adjustment
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Failed to load {os.path.basename(model_path)}: {e}")
        return None

    model.to(DEVICE)
    model.eval()

    # 3. Create Dummy Input
    dummy_input = torch.randn(INPUT_SHAPE).to(DEVICE)

    # 4. Warmup (crucial for GPU to initialize cuDNN benchmarks)
    # Run 10 passes that we don't count
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 5. Measure Time
    # Synchronize before starting timer (if using GPU)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    num_iterations = 100
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
            # Sync after every step isn't strictly necessary for throughput, 
            # but usually we sync at the very end for batch measurement.
            
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_iterations
    fps = 1.0 / avg_time_per_inference

    return {
        "Model": os.path.basename(model_path),
        "Avg_Inference_Time_sec": avg_time_per_inference,
        "FPS": fps
    }

def main():
    # Find all model files
    model_files = glob.glob(os.path.join(MODEL_FOLDER, "*.pth")) + \
                  glob.glob(os.path.join(MODEL_FOLDER, "*.pt"))
    
    if not model_files:
        print("No model files found!")
        return

    print(f"Found {len(model_files)} models. Starting measurement on {DEVICE}...")
    
    results = []
    
    for i, m_file in enumerate(model_files):
        print(f"[{i+1}/{len(model_files)}] Processing {os.path.basename(m_file)}...")
        stats = measure_inference_time(m_file)
        if stats:
            results.append(stats)

    # Save Results
    df = pd.DataFrame(results)
    df = df.sort_values(by="Avg_Inference_Time_sec")
    
    print("\n=== Results ===")
    print(df)
    
    df.to_csv("inference_benchmark_results.csv", index=False)
    print("\nSaved to inference_benchmark_results.csv")

if __name__ == "__main__":
    main()