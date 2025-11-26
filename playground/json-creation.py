import os
import torch
import json
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Directory containing the already merged .pt files
MERGED_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_predicted"

# Output JSON path
OUTPUT_JSON = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_dataset_from_pt.json"

def main():
    json_index = []
    
    # Find all .pt files in the merged directory
    pt_files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.pt")))
    print(f"Found {len(pt_files)} merged .pt files in {MERGED_DIR}")

    for pt_file in tqdm(pt_files, desc="Reading Merged Files"):
        try:
            # Load the .pt file
            data = torch.load(pt_file, map_location='cpu')
            
            # Extract lists/tensors
            # Ensure we handle potential missing keys gracefully
            if 'object_ids' not in data or len(data['object_ids']) == 0:
                continue

            object_ids = data['object_ids']
            
            # Handle labels (tensor -> list)
            if 'labels' in data:
                labels = data['labels'].tolist()
            else:
                # Fallback if labels aren't saved (unlikely for merged data)
                labels = [0] * len(object_ids)

            # Handle scores (tensor -> list)
            # NOTE: If your previous merge script didn't save 'scores', this will default to 1.0
            if 'scores' in data:
                scores = data['scores'].tolist()
            else:
                scores = [1.0] * len(object_ids)

            # Verification: Check if lengths match
            if not (len(object_ids) == len(labels) == len(scores)):
                print(f"Warning: Length mismatch in {os.path.basename(pt_file)}. Skipping...")
                continue

            # Aggregate into the JSON list
            for i in range(len(object_ids)):
                json_index.append({
                    "id": object_ids[i],
                    "cls": int(labels[i]),  # Ensure standard Python int
                    "scr": round(float(scores[i]), 4)
                })

        except Exception as e:
            print(f"Error processing {pt_file}: {e}")

    # --- Save Global JSON ---
    print(f"\nSaving JSON index ({len(json_index)} items) to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_index, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()