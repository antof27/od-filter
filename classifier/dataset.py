import torch
from torch.utils.data import Dataset
import glob
import os
import json
from tqdm import tqdm
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import glob
import os
import json
from tqdm import tqdm
import random
from collections import defaultdict

class ImageLevelDataset(Dataset):
    def __init__(self, pt_folder, json_file, num_pos=1, num_total=2):
        self.pt_folder = pt_folder
        self.num_pos = num_pos
        self.num_total = num_total
        
        print(f"Loading labels from {json_file}...")
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        self.target_map = {item['id']: item['cls'] for item in json_data}
        
        # We store info at the CLIP level, containing a list of valid frames inside
        self.valid_clips = [] 
        
        pt_files = glob.glob(os.path.join(pt_folder, "*.pt"))
        print(f"Scanning {len(pt_files)} clips...")
        
        for f_path in tqdm(pt_files, desc="Indexing Clips"):
            try:
                # Load lightweight header if possible, or full file
                data = torch.load(f_path, map_location='cpu', weights_only=True)
                obj_ids = data['object_ids'] 
                
                # frame_id -> {'pos': [indices], 'neg': [indices]}
                frames_in_clip = defaultdict(lambda: {'pos': [], 'neg': []})
                has_valid_data = False

                for idx, obj_id in enumerate(obj_ids):
                    if obj_id in self.target_map:
                        label = self.target_map[obj_id]
                        
                        # Parse Frame ID
                        try: frame_id = obj_id.rsplit('_', 1)[0]
                        except: continue

                        if label == 1: 
                            frames_in_clip[frame_id]['pos'].append(idx)
                        else: 
                            frames_in_clip[frame_id]['neg'].append(idx)
                            
                # Filter out frames that are empty (no pos AND no neg)
                valid_frames_metadata = []
                for fid, groups in frames_in_clip.items():
                    if len(groups['pos']) > 0 or len(groups['neg']) > 0:
                        valid_frames_metadata.append({
                            'id': fid,
                            'pos': groups['pos'],
                            'neg': groups['neg']
                        })
                
                # Only add the clip if it has at least one useful frame
                if len(valid_frames_metadata) > 0:
                    self.valid_clips.append({
                        'path': f_path,
                        'frames': valid_frames_metadata
                    })
                        
            except Exception as e:
                # print(f"Error reading {f_path}: {e}")
                pass
                
        print(f"Dataset Loaded. Found {len(self.valid_clips)} valid CLIPS containing training data.")

    def __len__(self):
        return len(self.valid_clips)

    def __getitem__(self, idx):
        clip_info = self.valid_clips[idx]
        
        # === OPTIMIZATION: LOAD FILE ONCE ===
        data = torch.load(clip_info['path'], map_location='cpu', weights_only=True)
        all_embeddings = data['embeddings']
        
        batch_embs = []
        batch_lbls = []
        
        # Iterate over ALL valid frames in this single clip
        for frame_meta in clip_info['frames']:
            available_pos = frame_meta['pos']
            available_neg = frame_meta['neg']
            
            # Sampling Logic (Same as before)
            n_pos_to_take = min(self.num_pos, len(available_pos))
            n_slots_remaining = self.num_total - n_pos_to_take
            n_neg_to_take = min(n_slots_remaining, len(available_neg))
            
            # Positive Samples
            if n_pos_to_take > 0:
                choices = random.sample(available_pos, k=n_pos_to_take)
                for i in choices:
                    batch_embs.append(all_embeddings[i]) # Keep as tensor ref
                    batch_lbls.append(1.0)
                    
            # Negative Samples
            if n_neg_to_take > 0:
                choices = random.sample(available_neg, k=n_neg_to_take)
                for i in choices:
                    batch_embs.append(all_embeddings[i])
                    batch_lbls.append(0.0)

        if len(batch_embs) == 0:
            return torch.zeros(0, 1280), torch.zeros(0)

        # Stack everything found in this clip
        return torch.stack(batch_embs).float(), torch.tensor(batch_lbls, dtype=torch.float)

def flatten_collate_fn(batch):
    embs, lbls = zip(*batch)
    embs_flat = torch.cat(embs, dim=0)
    lbls_flat = torch.cat(lbls, dim=0)
    return embs_flat, lbls_flat










class ImageLevelDatasetBoxes(Dataset):
    def __init__(self, pt_folder, json_file, num_pos=1, num_total=2):
        self.pt_folder = pt_folder
        self.num_pos = num_pos
        self.num_total = num_total

        # Validation
        if self.num_total < self.num_pos:
            raise ValueError(f"num_total ({num_total}) cannot be smaller than num_pos ({num_pos})")
        
        print(f"Loading labels from {json_file}...")
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        self.target_map = {item['id']: item['cls'] for item in json_data}
        self.valid_samples = [] 
        
        pt_files = glob.glob(os.path.join(pt_folder, "*.pt"))
        print(f"Scanning {len(pt_files)} files to index individual frames...")
        
        for f_path in tqdm(pt_files, desc="Indexing Frames (Boxes)"):
            try:
                # Load header/data to parse object IDs
                data = torch.load(f_path, map_location='cpu', weights_only=True)
                obj_ids = data['object_ids'] 
                
                # Group indices by Frame ID
                # Structure: frame_id -> {'pos': [idx1], 'neg': [idx2]}
                frames_in_clip = defaultdict(lambda: {'pos': [], 'neg': []})
                
                for idx, obj_id in enumerate(obj_ids):
                    if obj_id in self.target_map:
                        label = self.target_map[obj_id]
                        
                        # Parse Frame ID (remove the object index suffix)
                        # ID format: UUID_SEQ_FRAME_OBJ -> Frame ID: UUID_SEQ_FRAME
                        try:
                            frame_id = obj_id.rsplit('_', 1)[0]
                        except Exception:
                            continue

                        if label == 1: 
                            frames_in_clip[frame_id]['pos'].append(idx)
                        else: 
                            frames_in_clip[frame_id]['neg'].append(idx)
                
                # Append each valid FRAME as a sample
                for frame_id, groups in frames_in_clip.items():
                    if len(groups['pos']) > 0 or len(groups['neg']) > 0:
                        self.valid_samples.append({
                            'path': f_path,          # File path (Clip)
                            'frame_id': frame_id,    # Frame ID
                            'pos': groups['pos'],    # Indices for this frame
                            'neg': groups['neg']     # Indices for this frame
                        })
                        
            except Exception as e:
                # print(f"Error reading {f_path}: {e}")
                pass
                
        print(f"Dataset Loaded. Found {len(self.valid_samples)} valid frames.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_info = self.valid_samples[idx]
        data = torch.load(img_info['path'], map_location='cpu', weights_only=True)
        
        # These embeddings are 1284 dim (1280 emb + 4 box)
        all_embeddings = data['embeddings'] 
        
        selected_embs = []
        selected_lbls = []

        # Get indices specific to this FRAME
        available_pos = img_info['pos']
        available_neg = img_info['neg']
        
        # 1. Determine how many Positives to take
        n_pos_to_take = min(self.num_pos, len(available_pos))
        
        # 2. Determine how many Negatives to take
        n_slots_remaining = self.num_total - n_pos_to_take
        n_neg_to_take = min(n_slots_remaining, len(available_neg))
        
        # 3. Sampling
        if n_pos_to_take > 0:
            choices = random.sample(available_pos, k=n_pos_to_take)
            for i in choices:
                selected_embs.append(all_embeddings[i].float())
                selected_lbls.append(1.0)
                
        if n_neg_to_take > 0:
            choices = random.sample(available_neg, k=n_neg_to_take)
            for i in choices:
                selected_embs.append(all_embeddings[i].float())
                selected_lbls.append(0.0)
        
        # Error Handling
        if len(selected_embs) == 0:
             raise ValueError(
                f"No embeddings selected for index {idx}. "
                f"File: {img_info['path']}. "
                f"Avail Pos: {len(available_pos)}, Avail Neg: {len(available_neg)}. "
                f"Req Pos: {self.num_pos}, Req Total: {self.num_total}."
            )

        return torch.stack(selected_embs), torch.tensor(selected_lbls, dtype=torch.float)