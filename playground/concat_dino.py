import os
import torch
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_MERGED_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_predicted"
INPUT_DINO_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/dino_converted"

OUTPUT_CONCAT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concatenated_embeddings"

def load_dino_map(pt_path):
    """
    Loads a specific DINO .pt file for a clip and returns a lookup dict.
    Returns: { 'frame_filename': embedding_tensor }, embedding_dim
    """
    if not os.path.exists(pt_path):
        return None, 0

    try:
        data = torch.load(pt_path, map_location='cpu')
        
        # Structure check
        if 'object_ids' not in data or 'embeddings' not in data:
            return {}, 0

        ids = data['object_ids'] # In DINO files, these are frame filenames
        embs = data['embeddings']
        
        if len(embs) == 0:
            return {}, 0
            
        dim = embs[0].shape[0]
        
        # Create fast lookup dictionary
        dino_map = {ids[i]: embs[i] for i in range(len(ids))}
        return dino_map, dim

    except Exception as e:
        print(f"Error loading DINO file {pt_path}: {e}")
        return None, 0

def main():
    # Create output directory
    os.makedirs(OUTPUT_CONCAT_DIR, exist_ok=True)

    # Get list of merged .pt files
    merged_files = sorted(glob.glob(os.path.join(INPUT_MERGED_DIR, "*.pt")))
    print(f"Found {len(merged_files)} merged clips to process.")

    # We assume DINO files have the same filenames
    
    for merged_path in tqdm(merged_files, desc="Concatenating Clips"):
        filename = os.path.basename(merged_path)
        dino_path = os.path.join(INPUT_DINO_DIR, filename)
        
        # 1. Load DINO data for this specific clip
        dino_map, dino_dim = load_dino_map(dino_path)
        
        if dino_map is None:
            print(f"Skipping {filename}: Corresponding DINO file not found at {dino_path}")
            continue
            
        if dino_dim == 0:
            print(f"Warning: DINO file {filename} is empty or has 0 dim. Filling with zeros.")
            # Set an arbitrary dimension if unknown (e.g., 768 for Base, 1024 for Large)
            # You might want to hardcode this if empty files are common.
            continue

        zero_dino = torch.zeros(dino_dim)

        # 2. Load Merged Object data
        try:
            data = torch.load(merged_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading merged file {filename}: {e}")
            continue

        obj_embeddings = data['embeddings']
        image_ids = data['object_ids'] # These are the frame filenames (e.g. video_001.jpg)
        image_ids = [img.split("_")[0] for img in image_ids]
        
        new_embeddings_list = []
        
        # 3. Iterate through objects and concatenate
        num_objs = len(obj_embeddings)
        
        for i in range(num_objs):
            current_obj_emb = obj_embeddings[i]
            frame_name = image_ids[i] 
            
            # Lookup global embedding for this frame
            if frame_name in dino_map:
                current_dino_emb = dino_map[frame_name]
            else:
                # If frame is missing in DINO file (rare but possible mismatch)
                current_dino_emb = zero_dino
            
            # Concatenate: [Object] + [Global]
            # Ensure both are on CPU before cat
            concatenated = torch.cat((current_obj_emb, current_dino_emb), dim=0)
            new_embeddings_list.append(concatenated)

        # 4. Save
        if len(new_embeddings_list) > 0:
            final_data = {
                "object_ids": data['object_ids'],
                "embeddings": torch.stack(new_embeddings_list),
                "bboxes": data['bboxes'],
                "labels": data['labels']
            }
            
            output_path = os.path.join(OUTPUT_CONCAT_DIR, filename)
            torch.save(final_data, output_path)
        else:
            print(f"Warning: {filename} resulted in empty data.")

    print("\nProcessing complete.")
    print(f"Output saved to: {OUTPUT_CONCAT_DIR}")

if __name__ == "__main__":
    main()