import os
import torch
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Replace this with the path to your final concatenated folder
INPUT_FOLDER = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concatenated_embeddings"

def main():
    # 1. Get list of files
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.pt")))
    
    print(f"Scanning {len(files)} files in: {INPUT_FOLDER}")
    
    total_embeddings = 0
    files_with_errors = 0
    empty_files = 0

    # 2. Iterate and Count
    for file_path in tqdm(files, desc="Counting Embeddings"):
        try:
            # Load to CPU to avoid memory issues
            data = torch.load(file_path, map_location='cpu')
            
            # Check for 'embeddings' key
            if 'embeddings' in data:
                embeddings = data['embeddings']
                
                # Check if it's a tensor or list
                if isinstance(embeddings, torch.Tensor):
                    count = embeddings.shape[0]
                else:
                    count = len(embeddings)
                
                if count == 0:
                    empty_files += 1
                
                total_embeddings += count
            else:
                # Fallback if structure is different (e.g. just a tensor)
                if isinstance(data, torch.Tensor):
                    total_embeddings += data.shape[0]
                else:
                    print(f"Warning: No 'embeddings' key found in {os.path.basename(file_path)}")
                    
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
            files_with_errors += 1

    # 3. Print Results
    print("-" * 30)
    print(f"Total Files Scanned: {len(files)}")
    print(f"Files with 0 embeddings: {empty_files}")
    print(f"Files with errors:     {files_with_errors}")
    print("-" * 30)
    print(f"TOTAL EMBEDDINGS:    {total_embeddings}")
    print("-" * 30)

if __name__ == "__main__":
    main()