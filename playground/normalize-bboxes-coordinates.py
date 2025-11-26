import torch
import json
import os
import glob
from tqdm import tqdm

def normalize_and_save(json_path, input_folder, output_folder):
    # 1. Setup Output Directory
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Load JSON into a Lookup Dictionary
    # We map ID -> (width, height) for O(1) access speed
    print(f"Loading metadata from {json_path}...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Create the map
    # dim_map = { "id_string": (1920, 1080), ... }
    dim_map = {item['id']: (float(item['width']), float(item['height'])) for item in json_data}
    
    print(f"Metadata loaded. Found dimensions for {len(dim_map)} items.")

    # 3. Process .pt files
    pt_files = glob.glob(os.path.join(input_folder, "*.pt"))
    print(f"Found {len(pt_files)} .pt files to process.")

    for f_path in tqdm(pt_files, desc="Normalizing BBoxes"):
        try:
            # Load the file on CPU
            data = torch.load(f_path, map_location='cpu')
            
            # Extract relevant data
            object_ids = data['object_ids']
            bboxes = data['bboxes'] # Tensor of shape [N, 4]
            
            # Prepare tensors for scaling
            # We need to build a tensor of widths and heights that matches the bboxes shape
            widths = []
            heights = []
            valid_indices = [] # To keep track of which objects we successfully found dims for
            
            for i, obj_id in enumerate(object_ids):
                if obj_id in dim_map:
                    w, h = dim_map[obj_id]
                    widths.append(w)
                    heights.append(h)
                    valid_indices.append(i)
                else:
                    # Fallback logic: If ID not found in JSON
                    # Option A: Skip (but this changes tensor size vs other keys)
                    # Option B: Assume standard 1920x1080 (Risky)
                    # Option C: Print warning and use 1.0 (Effective NO-OP, but keeps data)
                    print(f"\n[Warning] ID {obj_id} not found in JSON. Skipping normalization for this object.")
                    widths.append(1.0) # Avoid division by zero
                    heights.append(1.0)
            
            # Convert to Tensor [N, 1]
            # Ensure dtype matches the original bboxes (usually float16 or float32)
            w_tensor = torch.tensor(widths, dtype=bboxes.dtype).unsqueeze(1)
            h_tensor = torch.tensor(heights, dtype=bboxes.dtype).unsqueeze(1)
            
            # Create Scale Tensor [N, 4] -> [w, h, w, h]
            # Because format is likely [x1, y1, x2, y2]
            scale_tensor = torch.cat([w_tensor, h_tensor, w_tensor, h_tensor], dim=1)
            
            # Perform Normalization
            # New coords will be between 0.0 and 1.0
            normalized_bboxes = bboxes / scale_tensor
            
            # Update the dictionary
            data['bboxes'] = normalized_bboxes
            
            # Save to new folder
            filename = os.path.basename(f_path)
            save_path = os.path.join(output_folder, filename)
            torch.save(data, save_path)
            
        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    print(f"Done! Processed files saved in: {output_folder}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # 1. Path to your JSON containing IDs, width, and height
    JSON_PATH = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_dataset_from_pt_with_dims.json"
    
    # 2. Folder containing the original .pt files
    INPUT_FOLDER = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concatenated_embeddings"
    
    # 3. New folder where modified files will be saved
    OUTPUT_FOLDER = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concatenated_embeddings_with_normalized_bboxes"
    
    normalize_and_save(JSON_PATH, INPUT_FOLDER, OUTPUT_FOLDER)