import sys
import torch
from autoencoder.encoder import VariationalEncoder

class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except:
            print('Encoder could not be initialized.')
            sys.exit()
    
    def process(self, observation):
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        new_observation = torch.cat((image_obs.view(-1), navigation_obs), -1)

        if len(observation) > 2:
            connectivity_obs = torch.tensor(observation[2], dtype=torch.float32).to(self.device)
            new_observation = torch.cat((new_observation, connectivity_obs.view(-1)), -1)

        remaining_zeros = torch.zeros(108 - new_observation.shape[0], device=self.device)
        new_observation = torch.cat((new_observation, remaining_zeros), dim=0)

        return new_observation