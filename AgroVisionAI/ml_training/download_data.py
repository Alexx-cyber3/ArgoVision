import os
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import shutil

def setup_expert_dataset():
    dataset_dir = Path("dataset")
    # Clean up old sample data to avoid mixing
    if dataset_dir.exists():
        print(f"Purging old dataset in {dataset_dir} for fresh expert download...")
        shutil.rmtree(dataset_dir)
        
    os.makedirs(dataset_dir, exist_ok=True)

    print("--- AgroVision AI Expert Dataset Downloader ---")
    print("Downloading FULL PlantVillage Dataset (54,303 images)...")
    print("This provides the 'Best Data' required for maximum accuracy.")
    
    # Download the full dataset
    ds_full, info = tfds.load('plant_village', with_info=True, as_supervised=True, split='train')
    
    label_names = info.features['label'].names
    
    # Save labels for the predictor
    import json
    labels_path = dataset_dir / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(label_names, f)
    print(f"Saved class labels to {labels_path}")

    total_images = info.splits['train'].num_examples
    
    print(f"Total images found: {total_images}")
    print("Organizing images into expert-level classification folders...")
    
    count = 0
    for image, label in ds_full:
        # Label format in PlantVillage is usually "Plant_Name__Disease_Name"
        raw_label = label_names[label]
        
        # Split into Part (mostly Leaf in PV) and Disease
        # We will structure it as: dataset/<Plant>/<Disease>
        if "__" in raw_label:
            plant_name, disease_name = raw_label.split("__")
        else:
            plant_name = "General"
            disease_name = raw_label
            
        class_path = dataset_dir / plant_name / disease_name
        os.makedirs(class_path, exist_ok=True)
        
        img_path = class_path / f"img_{count}.jpg"
        tf.keras.utils.save_img(str(img_path), image.numpy())
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count}/{total_images} images ({(count/total_images*100):.1f}%)...")

    print(f"
SUCCESS: Expert Dataset Ready at '{dataset_dir}'")
    print(f"Total images saved: {count}")
    print("
RECOMMENDATION: Run the enhanced 'train.py' now to achieve maximum accuracy.")

if __name__ == "__main__":
    setup_expert_dataset()
