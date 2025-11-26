import json
import random
from collections import defaultdict
import os

def split_dataset_three_ways(input_json_path, output_dir='./'):
    print(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # 1. Raggruppa per Clip
    clips_dict = defaultdict(list)
    for entry in data:
        # Assumendo format "clipname_frame_obj"
        clip_id = entry['id'].split('_')[0]
        clips_dict[clip_id].append(entry)

    unique_clips = list(clips_dict.keys())
    random.shuffle(unique_clips)
    
    total_clips = len(unique_clips)
    
    # 2. Definisci le percentuali
    # Test: 10%, Val: 1% (sufficiente visto il dataset enorme), Train: Resto
    test_count = int(total_clips * 0.10)
    val_count = int(total_clips * 0.01) # 1% per validation è veloce
    
    # 3. Slicing
    test_clips = unique_clips[:test_count]
    val_clips = unique_clips[test_count : test_count + val_count]
    train_clips = unique_clips[test_count + val_count:]

    print(f"Total Clips: {total_clips}")
    print(f"Train Clips: {len(train_clips)}")
    print(f"Val Clips:   {len(val_clips)}")
    print(f"Test Clips:  {len(test_clips)}")

    # 4. Helper per salvare
    def save_split(clips, filename):
        out_data = []
        for c in clips:
            out_data.extend(clips_dict[c])
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Saved {path}: {len(out_data)} samples")

    save_split(train_clips, '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/train_set.json')
    save_split(val_clips, '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/val_set.json')
    save_split(test_clips, '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/test_set.json')

if __name__ == "__main__":
    split_dataset_three_ways("/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/merged_dataset_from_pt.json")