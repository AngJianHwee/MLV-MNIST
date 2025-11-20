import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.latent_space = nn.Linear(256, 2)
        self.output_layer = nn.Linear(2, 10)

    def forward(self, x):
        x = self.flatten(x)
        features = self.layers(x)
        latent_representation = self.latent_space(features)
        output = self.output_layer(latent_representation)
        return output

    def get_latent_features(self, x):
        x = self.flatten(x)
        features = self.layers(x)
        latent_representation = self.latent_space(features)
        return latent_representation
