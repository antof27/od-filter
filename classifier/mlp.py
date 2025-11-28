import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast 
import os
import sys 


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, '..'))
data_dir = os.path.abspath(os.path.join(parent_dir, 'data'))
output_dir = os.path.abspath(os.path.join(parent_dir, 'weights'))


# === MODULAR IMPORTS ===
from model import EmbeddingMLP
from loss import FocalLoss
from utils import EarlyStopping
from dataset import ImageLevelDataset, flatten_collate_fn 




def train_model():
    PT_FOLDER = os.path.join(data_dir, 'concat_embeds')
    TRAIN_JSON = os.path.join(data_dir, 'train_set.json')
    VAL_JSON  = os.path.join(data_dir, 'val_set.json')
    

    # Since 1 item = 1 Clip (which might have 50 frames * 2 embeddings = 100 vectors),
    BATCH_SIZE = 32  
    EPOCHS = 10
    LR = 1e-4 # Increased slightly due to effective larger batch size
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    PATIENCE = 3

    train_dataset = ImageLevelDataset(PT_FOLDER, TRAIN_JSON, num_pos=1, num_total=2)
    val_dataset = ImageLevelDataset(PT_FOLDER, VAL_JSON, num_pos=1, num_total=2)

    # === OPTIMIZED DATALOADERS ===
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,         
        collate_fn=flatten_collate_fn,
        pin_memory=True,         # Fast CPU->GPU transfer
        persistent_workers=True, 
        prefetch_factor=2        
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, 
        collate_fn=flatten_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = EmbeddingMLP().to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Mixed Precision Scaler
    scaler = GradScaler() 
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=os.path.join(output_dir, 'best_mlp_model.pth'), mode='max')
    best_val_acc = 0.0

    print("Starting Training (Fast Mode)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for embeddings, labels in pbar:
            if embeddings.shape[0] == 0: continue

            embeddings, labels = embeddings.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # === MIXED PRECISION FORWARD PASS ===
            with autocast():
                outputs = model(embeddings).squeeze()
                if outputs.ndim == 0: outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, labels)

            # === MIXED PRECISION BACKWARD PASS ===
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{train_loss / (pbar.n + 1):.4f}"})

        epoch_train_acc = 100 * correct / total if total > 0 else 0
        
        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                if embeddings.shape[0] == 0: continue
                embeddings, labels = embeddings.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                with autocast():
                    outputs = model(embeddings).squeeze()
                    if outputs.ndim == 0: outputs = outputs.unsqueeze(0)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Ep {epoch+1}: Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        early_stopping(epoch_val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

if __name__ == "__main__":
    train_model()