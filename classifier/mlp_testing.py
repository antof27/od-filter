import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# IMPORT FROM YOUR TRAINING FILE
from classifier.mlp_boxes import EmbeddingMLP, ImageLevelDataset, flatten_collate_fn

def test_model():
    # --- CONFIG ---
    PT_FOLDER = '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/concatenated_embeddings' 
    TEST_JSON = '/storage/team/EgoTracksFull/v2/yolo-world-hooks/data/test_set.json'
    MODEL_PATH = 'best_mlp_model.pth' 
    
    BATCH_SIZE = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    print("Initializing Test Dataset...")
    # We use the Imported Dataset class
    test_dataset = ImageLevelDataset(PT_FOLDER, TEST_JSON, num_pos=1, num_neg=1)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, collate_fn=flatten_collate_fn)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    # We use the Imported Model class
    model = EmbeddingMLP().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Error: {MODEL_PATH} not found. Run training first.")
        return

    model.eval()
    
    # 3. Inference
    all_preds = []
    all_targets = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for embeddings, labels in tqdm(test_loader):
            embeddings = embeddings.to(DEVICE)
            
            outputs = model(embeddings).squeeze()
            probs = torch.sigmoid(outputs)
            
            preds = (probs > 0.5).float().cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)

    # 4. Metrics
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Class 0 (Neg)', 'Class 1 (Pos)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    print(f"(TN: {cm[0][0]} | FP: {cm[0][1]})")
    print(f"(FN: {cm[1][0]} | TP: {cm[1][1]})")

if __name__ == "__main__":
    test_model()