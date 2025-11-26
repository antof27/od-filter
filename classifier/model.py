import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.5, activation=nn.GELU()):
        super(MLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            activation,
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim=1280, hidden_dims=[1024, 512, 256], dropout_rate=0.5):
        super(EmbeddingMLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Dynamically create hidden layers
        for h_dim in hidden_dims:
            layers.append(MLPBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification head
        self.classifier = nn.Linear(current_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)