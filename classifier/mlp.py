import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# === MODULAR IMPORTS ===
from model import EmbeddingMLP
from loss import FocalLoss
from utils import EarlyStopping
from dataset import ImageLevelDataset, flatten_collate_fn

def train_model():
    # PATHS
    PT_FOLDER = '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concat-embeds' 
    TRAIN_JSON = '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/train_set.json'
    VAL_JSON   = '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/val_set.json'
    
    BATCH_SIZE = 1024
    EPOCHS = 200
    LR = 1e-5
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    train_dataset = ImageLevelDataset(PT_FOLDER, TRAIN_JSON, num_pos=1, num_neg=1)
    val_dataset = ImageLevelDataset(PT_FOLDER, VAL_JSON, num_pos=1, num_neg=1)

    print(f"Total Training Images Found: {len(train_dataset)}")
    print(f"Total Validation Images Found: {len(val_dataset)}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Batches per Epoch: {len(train_dataset) // BATCH_SIZE}")
    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, collate_fn=flatten_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, collate_fn=flatten_collate_fn)

    model = EmbeddingMLP().to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=10, verbose=True, path='/storage/team/EgoTracksFull/v2/yolo-world-hooks/weights/best_classic_mlp.pth', mode='max')

    print("Starting Training...")
    for epoch in range(EPOCHS):
            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            
            for embeddings, labels in pbar:
                # Safety check for empty batches (if dataset filtering yields empty results)
                if embeddings.shape[0] == 0:
                    continue

                embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(embeddings).squeeze()
                
                # Handle edge case where batch size is 1
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_loss = train_loss / (total / BATCH_SIZE + 1)
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

            epoch_train_acc = 100 * correct / total if total > 0 else 0
            
            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    if embeddings.shape[0] == 0:
                        continue

                    embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
                    
                    outputs = model(embeddings).squeeze()
                    if outputs.ndim == 0: outputs = outputs.unsqueeze(0)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate averages
            avg_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            print(f"Summary Ep {epoch+1}: Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")
            
            # --- EARLY STOPPING CHECK ---
            # Pass the Validation accuracy
            early_stopping(epoch_val_acc, model)
            
            if early_stopping.early_stop:
                
                print("Early stopping triggered. Training finished.")
                break

            print("Process Complete.")

if __name__ == "__main__":
    train_model()