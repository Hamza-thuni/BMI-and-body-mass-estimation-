import os
import sys
from pathlib import Path

def main():
    root = Path("d:/Fyp_project_root/Fyp_project_root (1)")
    img_dir = root / "data (1)" / "Celeb-FBI Dataset"
    
    if not img_dir.exists():
        print("Directory not found")
        return
        
    fnames = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    zero_weights = []
    bad_format = []
    
    for fname in fnames:
        name = os.path.splitext(fname)[0]
        parts = name.split("_")
        if len(parts) < 3:
            bad_format.append(fname)
            continue
            
        w_str = parts[2]
        if not w_str.endswith('w'):
            bad_format.append(fname)
            continue
            
        try:
            w = float(w_str[:-1])
            if w <= 0:
                zero_weights.append(fname)
        except:
            bad_format.append(fname)
            
    print(f"Total files: {len(fnames)}")
    print(f"Files with 0 or negative weight: {len(zero_weights)}")
    for f in zero_weights[:20]:
        print("  ", f)
        
    print(f"\nFiles with weird format: {len(bad_format)}")
    for f in bad_format[:10]:
        print("  ", f)

if __name__ == "__main__":
    main()
