"""
Export trained classifier models from Kaggle to deployment format

Run this script after training in the Kaggle notebook to prepare models for deployment.
"""
import torch
import shutil
from pathlib import Path
import zipfile

# Paths (adjust based on your Kaggle environment)
KAGGLE_MODELS_DIR = Path("/kaggle/working/models")
OUTPUT_ZIP = Path("/kaggle/working/deployment_models.zip")

# Emotion categories
CATEGORIES = ["Boredom", "Engagement", "Confusion", "Frustration"]


def export_models():
    """Export trained classifiers for deployment"""
    
    if not KAGGLE_MODELS_DIR.exists():
        print(f"Error: Models directory not found at {KAGGLE_MODELS_DIR}")
        return
    
    # Create temporary export directory
    export_dir = Path("/kaggle/working/export")
    export_dir.mkdir(exist_ok=True)
    classifiers_dir = export_dir / "classifiers"
    classifiers_dir.mkdir(exist_ok=True)
    
    print("Exporting trained classifiers...")
    exported_count = 0
    
    for category in CATEGORIES:
        category_clean = category.strip()
        
        # Look for MLP classifier (best performing)
        checkpoint_path = KAGGLE_MODELS_DIR / f"mlp_{category_clean}_best.pth"
        
        if checkpoint_path.exists():
            # Copy to export directory
            dest_path = classifiers_dir / f"mlp_{category_clean}_best.pth"
            shutil.copy2(checkpoint_path, dest_path)
            
            # Load and print info
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            val_f1 = checkpoint.get('val_f1', 'N/A')
            epoch = checkpoint.get('epoch', 'N/A')
            
            print(f"✓ Exported {category_clean}:")
            print(f"  - Epoch: {epoch}")
            print(f"  - Val F1: {val_f1:.4f if isinstance(val_f1, float) else val_f1}")
            print(f"  - Size: {dest_path.stat().st_size / 1024:.1f} KB")
            
            exported_count += 1
        else:
            print(f"✗ Not found: {category_clean}")
    
    print(f"\nExported {exported_count}/{len(CATEGORIES)} classifiers")
    
    # Create README
    readme_path = export_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("Trained Emotion Classifiers\n")
        f.write("=" * 50 + "\n\n")
        f.write("Installation:\n")
        f.write("1. Extract this zip file\n")
        f.write("2. Copy 'classifiers/' folder to 'deployment/models/'\n")
        f.write("3. Start the backend server\n\n")
        f.write(f"Exported: {exported_count} classifier models\n")
        f.write(f"Categories: {', '.join(CATEGORIES)}\n")
    
    # Create zip file
    print(f"\nCreating zip archive: {OUTPUT_ZIP}")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in export_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(export_dir)
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    print(f"\n✓ Export complete!")
    print(f"✓ Download: {OUTPUT_ZIP}")
    print(f"✓ Archive size: {OUTPUT_ZIP.stat().st_size / (1024**2):.1f} MB")
    
    # Cleanup
    shutil.rmtree(export_dir)
    print("\nNext steps:")
    print("1. Download the zip file from Kaggle")
    print("2. Extract to deployment/models/")
    print("3. Run the backend server")


if __name__ == "__main__":
    export_models()
