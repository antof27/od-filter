import torch

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys


from utils import EarlyStopping
from loss import FocalLoss
from model_wboxes import BoxAwareModel
from dataset import ImageLevelDatasetBoxes, flatten_collate_fn
import os 
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, '..'))
data_dir = os.path.abspath(os.path.join(parent_dir, 'data'))


def train_model():
    # PATHS
    PT_FOLDER = os.path.join(data_dir, 'concat_embeds')
    TRAIN_JSON = os.path.join(data_dir, 'train_set.json')
    VAL_JSON  = os.path.join(data_dir, 'val_set.json')
    
    BATCH_SIZE = 32
    EPOCHS = 10     
    PATIENCE = 3    
    LR = 1e-5
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    # Datasets
    train_dataset = ImageLevelDatasetBoxes(PT_FOLDER, TRAIN_JSON, num_pos=1, num_neg=1)
    val_dataset = ImageLevelDatasetBoxes(PT_FOLDER, VAL_JSON, num_pos=1, num_neg=1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, collate_fn=flatten_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, collate_fn=flatten_collate_fn)

    # Model Initialization
    model = BoxAwareModel(embedding_dim=1280, hidden_dim=512).to(DEVICE)
    
    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path='best_box_aware_model.pth', mode='max')

    print(f"Starting Training with Early Stopping (Patience={PATIENCE})...")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': train_loss / (total/BATCH_SIZE + 1)})

        train_acc = 100 * correct / total
        
        # --- VALIDATE ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
                outputs = model(embeddings).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1} Results | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # --- CALL EARLY STOPPING ---
        early_stopping(val_acc, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training Complete. Loading best model...")
    model.load_state_dict(torch.load('best_box_aware_model.pth'))

if __name__ == "__main__":
    train_model()