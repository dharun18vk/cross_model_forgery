import os
import shutil
from pathlib import Path

def simple_combine_datasets():
    # Define your source datasets and target directory
    source_datasets = [
        "real-vs-fake",
        "FFPP_Frames", 
        "CelebDF_Frames",
        "Dataset_deepfake"
    ]
    
    target_dir = "combined_dataset"
    
    # Create target structure
    splits = ['train', 'val']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)
    
    # Combine datasets
    for dataset_path in source_datasets:
        dataset_name = os.path.basename(dataset_path)
        
        for split in splits:
            for cls in classes:
                source_folder = os.path.join(dataset_path, split, cls)
                target_folder = os.path.join(target_dir, split, cls)
                
                if os.path.exists(source_folder):
                    for filename in os.listdir(source_folder):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            # Create unique filename
                            new_filename = f"{dataset_name}_{filename}"
                            source_file = os.path.join(source_folder, filename)
                            target_file = os.path.join(target_folder, new_filename)
                            
                            # Copy file
                            shutil.copy2(source_file, target_file)
                            print(f"Copied: {source_file} -> {target_file}")

if __name__ == "__main__":
    simple_combine_datasets()