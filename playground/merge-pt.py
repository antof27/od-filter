import os
import torch
import glob
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
PRED_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/yolo_predicted_converted"
GT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/yolo_gt_converted"

# Output Directory for merged .pt files
OUTPUT_PT_DIR = "/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_predicted"

IOU_THRESHOLD = 0.7

def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]."""
    b1 = box1.tolist() if isinstance(box1, torch.Tensor) else box1
    b2 = box2.tolist() if isinstance(box2, torch.Tensor) else box2

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    if union == 0: return 0.0
    return intersection / union

def load_and_group_by_image(pt_path):
    """Loads a .pt file and groups objects by image filename."""
    if not os.path.exists(pt_path):
        return {}
        
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return {}

    grouped = defaultdict(list)
    
    if 'object_ids' not in data or len(data['object_ids']) == 0:
        return {}

    num_objs = len(data['object_ids'])
    
    for i in range(num_objs):
        obj_id = data['object_ids'][i]
        parent_img = data['parent_image'][i] if 'parent_image' in data else obj_id.rsplit('_', 1)[0]

        obj_data = {
            'original_id': obj_id,
            'embedding': data['embeddings'][i],
            'bbox': data['bboxes'][i],
            'score': data['scores'][i] if 'scores' in data else 1.0,
        }
        grouped[parent_img].append(obj_data)
        
    return grouped

def main():
    os.makedirs(OUTPUT_PT_DIR, exist_ok=True)
    
    # 1. Identify all unique filenames involved (Union of Preds and GTs)
    pred_files = {os.path.basename(f) for f in glob.glob(os.path.join(PRED_DIR, "*.pt"))}
    gt_files = {os.path.basename(f) for f in glob.glob(os.path.join(GT_DIR, "*.pt"))}
    
    all_filenames = sorted(list(pred_files | gt_files))
    
    print(f"Found {len(pred_files)} prediction files and {len(gt_files)} GT files.")
    print(f"Total unique clips to process: {len(all_filenames)}")
    print(f"Outputting merged .pt files to: {OUTPUT_PT_DIR}")

    for filename in tqdm(all_filenames, desc="Processing Clips"):
        pred_path = os.path.join(PRED_DIR, filename)
        gt_path = os.path.join(GT_DIR, filename)
        
        # --- Reset buffers for THIS CLIP only ---
        clip_embeddings = []
        clip_bboxes = []
        clip_ids = []
        clip_labels = []
        clip_scores = [] 

        # Load data (returns empty dict if file doesn't exist)
        preds_by_image = load_and_group_by_image(pred_path) if filename in pred_files else {}
        gts_by_image = load_and_group_by_image(gt_path) if filename in gt_files else {}

        all_images = set(preds_by_image.keys()) | set(gts_by_image.keys())
        
        for img_name in all_images:
            img_preds = preds_by_image.get(img_name, [])
            img_gts = gts_by_image.get(img_name, [])

            # --- CASE A: Prediction exists (Perform Logic) ---
            if img_preds:
                # 1. Find max prediction index (for unique GT IDs later)
                max_pred_index = 0
                for p in img_preds:
                    try:
                        idx = int(p['original_id'].rsplit('_', 1)[-1])
                        if idx > max_pred_index: max_pred_index = idx
                    except ValueError: pass
                
                # 2. Process Predictions
                for p in img_preds:
                    # Check if this prediction matches any GT
                    is_match = False
                    if img_gts: # Only calculate IoU if GTs exist
                        for g in img_gts:
                            if calculate_iou(p['bbox'], g['bbox']) >= IOU_THRESHOLD:
                                is_match = True
                                break
                    
                    # Logic: 
                    # If matched (IoU >= 0.7) -> Label 1
                    # If not matched (IoU < 0.7) -> Label 0
                    # If no GT exists at all for this image -> Label 0
                    label = 1 if is_match else 0
                    
                    clip_embeddings.append(p['embedding'])
                    clip_bboxes.append(p['bbox'])
                    clip_ids.append(p['original_id'])
                    clip_scores.append(p['score'])
                    clip_labels.append(label)

                # 3. Add Ground Truths (Always Keep -> Label 1)
                for i, g in enumerate(img_gts):
                    # Create new ID to avoid conflict with predictions
                    new_id = f"{img_name}_{max_pred_index + i + 1:02d}"
                    
                    clip_embeddings.append(g['embedding'])
                    clip_bboxes.append(g['bbox'])
                    clip_ids.append(new_id)
                    clip_scores.append(g['score'])
                    clip_labels.append(1)

            # --- CASE B: No Prediction, Only Ground Truth ---
            else:
                # If there are no predictions, we simply add all GTs.
                # Since they are GTs, they are all Label 1.
                for i, g in enumerate(img_gts):
                    # ID handling: Use original or generate new?
                    # Since there are no preds, we can stick to a simple schema or keep original if available.
                    # To be consistent with logic above, we treat max_pred_index as 0.
                    new_id = f"{img_name}_{i + 1:02d}"
                    
                    clip_embeddings.append(g['embedding'])
                    clip_bboxes.append(g['bbox'])
                    clip_ids.append(new_id)
                    clip_scores.append(g['score'])
                    clip_labels.append(1)

        # --- Save Per-Clip PT File ---
        if len(clip_embeddings) > 0:
            final_pt_data = {
                "object_ids": clip_ids,
                "embeddings": torch.stack(clip_embeddings).cpu(),
                "bboxes": torch.stack(clip_bboxes).cpu(),
                "labels": torch.tensor(clip_labels, dtype=torch.long),
                "scores": torch.tensor(clip_scores, dtype=torch.float32) 
            }
            
            output_path = os.path.join(OUTPUT_PT_DIR, filename)
            torch.save(final_pt_data, output_path)

    print("Done. Merged .pt files saved.")

if __name__ == "__main__":
    main()