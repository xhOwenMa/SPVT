import torch
import torch.nn as nn

    
class CombinedNetwork(nn.Module):
    def __init__(self, generator, controller):
        super(CombinedNetwork, self).__init__()
        self.generator = generator
        self.controller = controller
        
    def generator_forward(self, z, c):
        return self.generator(z, c)

    def forward(self, z, c):
        gen_output = self.generator(z, c)
        control_output = self.controller(gen_output)
        return control_output

class E2ENetworkBN(nn.Module):
    def __init__(self, layer_sizes=[128, 64, 32, 8, 1]):
        """
        old version: [128, 50, 50, 50, 50, 1]
        new: [128, 64, 32, 8, 1]
        """
        super(E2ENetworkBN, self).__init__()
        layers = []
        for i in range(1, len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class E2ENetwork(nn.Module):
    def __init__(self, layer_sizes=[128, 64, 32, 8, 1]):
        """
        old version: [128, 50, 50, 50, 50, 1]
        new: [128, 64, 32, 8, 1]
        """
        super(E2ENetwork, self).__init__()
        layers = []
        for i in range(1, len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
        
    
class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=10, state_dim=2):
        super(MLPGenerator, self).__init__()

        self.input_dim = latent_dim + state_dim
        output_size = 8 * 16
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, output_size),
            nn.Tanh()
        )

    @staticmethod
    def sample_z(batch_size, dim_z, device='cpu'):
        """Sample latent vectors from uniform distribution [-1, 1]"""
        return 0.8 * (2 * torch.rand(batch_size, dim_z, device=device) - 1)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        return self.model(x)
    
def get_model():
    generator = MLPGenerator()
    controller = E2ENetworkBN()
    model = CombinedNetwork(generator, controller)
    return model