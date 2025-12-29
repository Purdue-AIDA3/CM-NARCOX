import torch.nn as nn

class FCHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LeakyReLU(0.2),
                    nn.LayerNorm(embedding_dim),
                    nn.Dropout(0.2),
                    nn.Linear(embedding_dim, embedding_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.LayerNorm(embedding_dim * 2),
                    nn.Linear(embedding_dim * 2, 1)
                )
        # self.quantiles = quantiles
        # self.fc = nn.Linear(embedding_dim, len(self.quantiles))
        self.initialize_fc_weights()

    def initialize_fc_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.fc(x)