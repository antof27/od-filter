import json
import os
from tqdm import tqdm

def create_dimension_lookup(coco_json_path):
    """
    Reads the COCO json and creates a fast lookup dictionary.
    Key: Image filename WITHOUT extension (e.g., "uuid_frame_videoframe")
    Value: {'width': w, 'height': h}
    """
    print(f"Loading COCO data from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # We assume the structure is standard COCO: {'images': [...], ...}
    images_list = coco_data.get('images', [])
    
    dim_map = {}
    print(f"Building Dimension Map from {len(images_list)} images...")
    
    for img in tqdm(images_list, desc="Indexing COCO"):
        file_name = img['file_name']
        width = img['width']
        height = img['height']
        
        # KEY LOGIC: Remove the extension (.jpg) to match the ID format later
        # Example: "abc..._123.jpg" -> "abc..._123"
        key_name = os.path.splitext(file_name)[0]
        
        dim_map[key_name] = {'width': width, 'height': height}
        
    return dim_map

def enrich_dataset(target_json_path, dim_map, output_path):
    """
    Loads the target dataset, looks up dimensions using the ID, 
    and saves the enriched version.
    """
    print(f"Processing {target_json_path}...")
    with open(target_json_path, 'r') as f:
        dataset = json.load(f)
    
    updated_count = 0
    missing_count = 0
    
    for item in tqdm(dataset, desc="Updating items"):
        obj_id = item['id']
        
        # KEY LOGIC: 
        # The object ID is: "UUID_Frame_VideoFrame_ObjID" (e.g., ..._10661_01)
        # The COCO key is:  "UUID_Frame_VideoFrame"       (e.g., ..._10661)
        # We need to remove the last part (the object suffix)
        
        # rsplit('_', 1) splits the string from the right, exactly once.
        # [0] takes the left part (the image ID)
        image_key = obj_id.rsplit('_', 1)[0]
        
        if image_key in dim_map:
            dims = dim_map[image_key]
            item['width'] = dims['width']
            item['height'] = dims['height']
            updated_count += 1
        else:
            missing_count += 1
            # Optional: Print first missing example to debug
            if missing_count == 1:
                print(f"\n[Warning] Could not find dimensions for ID: {obj_id} (Key searched: {image_key})")

    print(f"Finished. Updated {updated_count} items. Missing keys: {missing_count}")
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Done.\n")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Path to your COCO JSON (The one with width/height)
    COCO_PATH = "/storage/team/EgoTracksFull/v2/egotracks/train_dataset_versions/training_journal.json" 
    
    # 2. Paths to your Split JSONs (The ones to update)
    # You can add train, val, test here
    TARGET_FILES = [
        ("/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_dataset_from_pt.json", "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_dataset_from_pt_with_dims.json")
    ]
    
    # Check if COCO file exists
    if not os.path.exists(COCO_PATH):
        print("Error: COCO file not found. Please set COCO_PATH correctly.")
    else:
        # Step 1: Build the map once (it's heavy)
        dimension_map = create_dimension_lookup(COCO_PATH)
        
        # Step 2: Update all your files
        for input_file, output_file in TARGET_FILES:
            if os.path.exists(input_file):
                enrich_dataset(input_file, dimension_map, output_file)
            else:
                print(f"Skipping {input_file} (File not found)")