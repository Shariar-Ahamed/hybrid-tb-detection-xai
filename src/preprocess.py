import os
import pandas as pd
from pathlib import Path

def generate_metadata():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    normal_dir = raw_dir / "Normal"
    tb_dir = raw_dir / "Tuberculosis"
    
    print("=" * 60)
    print("           Tuberculosis CXR Metadata Generator")
    print("=" * 60)
    
    # Check directory existence
    if not normal_dir.exists() or not tb_dir.exists():
        print(f"[-] Error: Could not find Normal or Tuberculosis directories.")
        print(f"    Expected: {normal_dir}")
        print(f"    Expected: {tb_dir}")
        return False
        
    records = []
    
    # Scan Normal scans (labeled 0)
    print("[+] Scanning Normal CXR scans...")
    normal_count = 0
    for img_path in normal_dir.glob("*.png"):
        records.append({
            "image_path": f"raw/Normal/{img_path.name}",
            "label": 0,
            "class_name": "Normal"
        })
        normal_count += 1
        
    # Scan TB scans (labeled 1)
    print("[+] Scanning Tuberculosis CXR scans...")
    tb_count = 0
    for img_path in tb_dir.glob("*.png"):
        records.append({
            "image_path": f"raw/Tuberculosis/{img_path.name}",
            "label": 1,
            "class_name": "Tuberculosis"
        })
        tb_count += 1
        
    # Build dataframe
    df = pd.DataFrame(records)
    
    # Shuffle the dataset to ensure random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save CSV
    metadata_csv = data_dir / "metadata.csv"
    df.to_csv(metadata_csv, index=False)
    
    print("-" * 60)
    print(f"[+] Total Normal scans found: {normal_count}")
    print(f"[+] Total Tuberculosis scans found: {tb_count}")
    print(f"[+] Metadata CSV successfully generated at: {metadata_csv}")
    print(f"[+] Total entries in dataset: {len(df)}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    generate_metadata()
