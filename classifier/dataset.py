import torch
from torch.utils.data import Dataset
import glob
import os
import json
from tqdm import tqdm
import random

class ImageLevelDataset(Dataset):
    def __init__(self, pt_folder, json_file, num_pos=1, num_neg=1):
        self.pt_folder = pt_folder
        self.num_pos = num_pos
        self.num_neg = num_neg
        
        print(f"Loading labels from {json_file}...")
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        self.target_map = {item['id']: item['cls'] for item in json_data}
        self.valid_images = []
        
        pt_files = glob.glob(os.path.join(pt_folder, "*.pt"))
        print(f"Scanning {len(pt_files)} files against targets...")
        
        for f_path in tqdm(pt_files, desc="Indexing Dataset"):
            try:
                # Optimization: parse filename if possible, otherwise load header
                # If filenames contain IDs, parse them here to avoid torch.load
                data = torch.load(f_path, map_location='cpu', weights_only=True)
                obj_ids = data['object_ids'] 
                
                pos_indices = []
                neg_indices = []
                
                for idx, obj_id in enumerate(obj_ids):
                    if obj_id in self.target_map:
                        label = self.target_map[obj_id]
                        if label == 1: pos_indices.append(idx)
                        else: neg_indices.append(idx)
                
                if len(pos_indices) > 0 or len(neg_indices) > 0:
                    self.valid_images.append({
                        'path': f_path, 'pos': pos_indices, 'neg': neg_indices
                    })
                        
            except Exception as e:
                print(f"Error reading {f_path}: {e}")
                
        print(f"Dataset Loaded. Found {len(self.valid_images)} valid clips.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        data = torch.load(img_info['path'], map_location='cpu', weights_only=True)
        all_embeddings = data['embeddings']
        
        selected_embs = []
        selected_lbls = []
        
        if len(img_info['pos']) > 0:
            choices = random.choices(img_info['pos'], k=self.num_pos)
            for i in choices:
                selected_embs.append(all_embeddings[i].float())
                selected_lbls.append(1.0)
                
        if len(img_info['neg']) > 0:
            choices = random.choices(img_info['neg'], k=self.num_neg)
            for i in choices:
                selected_embs.append(all_embeddings[i].float())
                selected_lbls.append(0.0)
        
        if len(selected_embs) == 0:
            return torch.zeros(1, 1280), torch.zeros(1)

        return torch.stack(selected_embs), torch.tensor(selected_lbls, dtype=torch.float)

def flatten_collate_fn(batch):
    embs, lbls = zip(*batch)
    embs_flat = torch.cat(embs, dim=0)
    lbls_flat = torch.cat(lbls, dim=0)
    return embs_flat, lbls_flat






class ImageLevelDatasetBoxes(Dataset):
    def __init__(self, pt_folder, json_file, num_pos=1, num_neg=1):
        self.pt_folder = pt_folder
        self.num_pos = num_pos
        self.num_neg = num_neg
        
        print(f"Loading labels from {json_file}...")
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            
        self.target_map = {item['id']: item['cls'] for item in json_data}
        self.valid_images = []
        
        pt_files = glob.glob(os.path.join(pt_folder, "*.pt"))
        print(f"Scanning {len(pt_files)} files...")
        
        for f_path in tqdm(pt_files, desc="Indexing"):
            try:
                # Optimized: We assume filename implies content to skip loading if possible
                # But for safety we load header or handle error
                data = torch.load(f_path, map_location='cpu', weights_only=True)
                obj_ids = data['object_ids'] 
                
                pos_indices = []
                neg_indices = []
                
                for idx, obj_id in enumerate(obj_ids):
                    if obj_id in self.target_map:
                        label = self.target_map[obj_id]
                        if label == 1: pos_indices.append(idx)
                        else: neg_indices.append(idx)
                
                if len(pos_indices) > 0 or len(neg_indices) > 0:
                    self.valid_images.append({'path': f_path, 'pos': pos_indices, 'neg': neg_indices})
                        
            except Exception as e:
                pass
                
        print(f"Dataset Loaded. Found {len(self.valid_images)} valid images.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        data = torch.load(img_info['path'], map_location='cpu', weights_only=True)
        
        # These embeddings are 1284 dim (1280 emb + 4 box)
        all_embeddings = data['embeddings'] 
        
        selected_embs = []
        selected_lbls = []
        
        if len(img_info['pos']) > 0:
            choices = random.choices(img_info['pos'], k=self.num_pos)
            for i in choices:
                selected_embs.append(all_embeddings[i].float())
                selected_lbls.append(1.0)
                
        if len(img_info['neg']) > 0:
            choices = random.choices(img_info['neg'], k=self.num_neg)
            for i in choices:
                selected_embs.append(all_embeddings[i].float())
                selected_lbls.append(0.0)
        
        if len(selected_embs) == 0:
            return torch.zeros(1, 1284), torch.zeros(1)

        return torch.stack(selected_embs), torch.tensor(selected_lbls, dtype=torch.float)

def flatten_collate_fn(batch):
    embs, lbls = zip(*batch)
    embs_flat = torch.cat(embs, dim=0)
    lbls_flat = torch.cat(lbls, dim=0)
    return embs_flat, lbls_flat