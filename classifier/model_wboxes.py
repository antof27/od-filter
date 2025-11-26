import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(Block, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class BoxProjector(nn.Module):
    """ Project normalized box coords (4) to Embedding Dim (1280) """
    def __init__(self, output_dim=1280):
        super(BoxProjector, self).__init__()
        # We pass dropout=0.0 for coordinate projection
        self.layer1 = Block(4, 256, dropout=0.0)
        self.layer2 = Block(256, output_dim, dropout=0.0)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BoxAwareModel(nn.Module):
    def __init__(self, embedding_dim=1280, hidden_dim=512, dropout_rate=0.5):
        super(BoxAwareModel, self).__init__()
        
        # Branch 1: Project BBoxes
        self.pos_mlp = BoxProjector(output_dim=embedding_dim)
        
        # Branch 2: Main Classifier
        self.layer1 = Block(embedding_dim, hidden_dim, dropout_rate)
        self.layer2 = Block(hidden_dim, hidden_dim // 2, dropout_rate)
        self.head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x):
        # 1. SPLIT Inputs
        image_emb = x[:, :1280]  # First 1280 features
        box_coords = x[:, 1280:] # Last 4 features
        
        # 2. PROJECT Geometry
        pos_emb = self.pos_mlp(box_coords) 
        
        # 3. SUM (Fuse Semantic + Spatial)
        combined_features = image_emb + pos_emb
        
        # 4. CLASSIFY
        x = self.layer1(combined_features)
        x = self.layer2(x)
        x = self.head(x)
        
        return x