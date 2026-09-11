#!/usr/bin/env python3
"""Split the new dataset into train/valid and train the YOLO model.

The new datasets folder only has a train/ split. This script:
1. Splits train/ into train/ (80%) and valid/ (20%)
2. Updates data.yaml with correct absolute paths
3. Trains YOLO11n using transfer learning from the existing model
4. Exports the best model to NCNN format
"""
import os
import shutil
import random
import sys

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "new datasets")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sitophilus_oryzae_v2-3_best.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

def split_dataset():
    """Split train/ into train/ (80%) and valid/ (20%)."""
    images_dir = os.path.join(TRAIN_DIR, "images")
    labels_dir = os.path.join(TRAIN_DIR, "labels")

    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if not all_images:
        print("No images found in train/images!")
        return False

    print(f"Found {len(all_images)} images in train/")

    # Create valid/ directories
    valid_images_dir = os.path.join(VALID_DIR, "images")
    valid_labels_dir = os.path.join(VALID_DIR, "labels")
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    valid_images = all_images[split_idx:]

    print(f"Split: {len(train_images)} train, {len(valid_images)} valid")

    # Move valid images and labels
    moved = 0
    for img_file in valid_images:
        # Move image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(valid_images_dir, img_file)
        shutil.move(src_img, dst_img)

        # Move corresponding label (same name, .txt extension)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(valid_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

        moved += 1

    print(f"Moved {moved} images to valid/")
    return True

def update_data_yaml():
    """Update data.yaml with correct absolute paths."""
    content = f"""train: {os.path.join(TRAIN_DIR, 'images')}
val: {os.path.join(VALID_DIR, 'images')}
test: {os.path.join(TRAIN_DIR, 'images')}

nc: 1
names: ['Sitophilus-Oryzae']
"""
    with open(DATA_YAML, 'w') as f:
        f.write(content)
    print(f"Updated {DATA_YAML}")

def train_model():
    """Train YOLO11n using transfer learning from the existing model."""
    from ultralytics import YOLO

    print(f"\nLoading base model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Training with data: {DATA_YAML}")
    print("Training on CPU (no CUDA available) - this may take a while...")

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,
        device='cpu',
        project=OUTPUT_DIR,
        name='sitophilus_oryzae_v3',
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\nTraining complete!")
    print(f"Results: {results}")

    # Find the best model
    best_path = os.path.join(OUTPUT_DIR, 'sitophilus_oryzae_v3', 'weights', 'best.pt')
    if os.path.exists(best_path):
        print(f"Best model: {best_path}")
        return best_path
    else:
        print(f"Best model not found at {best_path}")
        return None

def export_ncnn(model_path):
    """Export the trained model to NCNN format."""
    from ultralytics import YOLO

    if model_path is None or not os.path.exists(model_path):
        print("No model to export!")
        return False

    print(f"\nExporting {model_path} to NCNN...")
    model = YOLO(model_path)
    ncnn_path = model.export(format='ncnn', imgsz=640)
    print(f"NCNN export: {ncnn_path}")
    return True

def main():
    print("=== Anilag YOLO Model Retraining ===\n")

    # Step 1: Split dataset
    print("--- Step 1: Splitting dataset ---")
    if os.path.exists(VALID_DIR) and os.listdir(os.path.join(VALID_DIR, "images", "")) if os.path.exists(os.path.join(VALID_DIR, "images")) else False:
        print("Valid split already exists, skipping split")
    else:
        if not split_dataset():
            print("Failed to split dataset!")
            sys.exit(1)

    # Step 2: Update data.yaml
    print("\n--- Step 2: Updating data.yaml ---")
    update_data_yaml()

    # Step 3: Train model
    print("\n--- Step 3: Training model ---")
    model_path = train_model()

    # Step 4: Export to NCNN
    print("\n--- Step 4: Exporting to NCNN ---")
    if not export_ncnn(model_path):
        print("NCNN export failed!")
        sys.exit(1)

    print("\n=== Done! ===")
    print(f"Trained model: {model_path}")
    print(f"NCNN model in: {os.path.dirname(model_path) if model_path else 'N/A'}")

if __name__ == "__main__":
    main()
