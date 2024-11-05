import os
import torch
import re 
import random
import numpy as np
from Segment import Segment
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# from RewardFunctions import linear_factor, logarithmic_factor_40, logarithmic_factor_60, logarithmic_factor_90, dumb_reward_scheme
# from fit_exp_curve import exp_decay, params25, params50, params75, params100
# from fit_polynomial_curve import poly_fit, coefs25, coefs50, coefs75, coefs100

class NetworkSelectionEnvImages:
    def __init__(self, image_folder, label_folder, performance_factor, device, model_paths, resize_dim=(84, 84), inference_dim=(512, 512), verbose=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.device = device
        self.verbose = verbose
        self.n_actions = 4
        self.current_idx = 0
        self.PERFORMANCE_FACTOR = performance_factor
        self.network_performance_columns = ['0%', '25%', '50%', '75%']
        self.resize_dim = resize_dim

        # Get and sort list of image filenames based on numeric parts for sequential processing
        self.image_filenames = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )

        # Set up inference models using Segment for each network level
        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_folder, label_dir=label_folder, resolution=inference_dim, device=device)
            for i in range(self.n_actions)
        }

        # Transform to downscale images to 84x84 for DQN input
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(),
        ])

        self.battery = 100
        
        # For 497 images
        self.energy_costs = { '0%': 0.5, '25%': 0.25, '50%': 0.125, '75%': 0.0625 }

        self.reset()

    def load_image(self, filename):
        """Load and preprocess an image."""
        image_path = os.path.join(self.image_folder, filename)
        image = Image.open(image_path).convert('RGB')  # Ensure RGB
        image_tensor = self.transform(image)
        return self.transform(image).to(self.device)
 
    def reset(self):
        self.current_idx = 0
        self.battery = 100  
        image_tensor = self.load_image(self.image_filenames[self.current_idx])
        state = self.get_state(image_tensor)
        return state

    def get_state(self, image_tensor):
        # Normalize battery level and images left
        battery_level = torch.tensor([self.battery / 100.0], device=self.device)
        images_left = torch.tensor([(len(self.image_filenames) - self.current_idx) / len(self.image_filenames)], device=self.device)

        # Stack as additional channels
        state = torch.cat((image_tensor, battery_level.unsqueeze(0).unsqueeze(2).expand(-1, 84, 84),
                           images_left.unsqueeze(0).unsqueeze(2).expand(-1, 84, 84)), dim=0)       
        return state

    def sample_action(self):
        """Randomly select an action."""
        return random.randint(0, self.n_actions - 1)

    def get_next_state_reward(self, action):
        """Determine the next state and reward after taking an action."""
        filename = self.image_filenames[self.current_idx]

        selected_network = self.network_performance_columns[action]
        # Perform inference to get IoU score for the selected network level
        results, _, _, _ = self.segments[action].infer(filename)

        selected_network_iou = results['test/iou/weeds']

        # Reward scheme 3: using weighted sum
        reward = (selected_network_iou * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * (1 - self.energy_costs[selected_network])

        self.battery -= self.energy_costs[selected_network]

        if self.verbose:
            print(f"Filename: {filename}")
            print(f"Selected Network: {selected_network}")
            print(f"IoU: {selected_network_iou}")
            print(f"Battery Level: {self.battery}")

        # Move to the next image
        self.current_idx = (self.current_idx + 1) % len(self.image_filenames)
        done = self.current_idx == 0

        next_image_tensor = self.load_image(self.image_filenames[self.current_idx])
        next_state = self.get_state(next_image_tensor)

        if self.battery <= 0:
            return next_state, float('-inf'), selected_network_iou, done 

        return next_state, reward, selected_network_iou, done

    def has_next(self):
        """Check if there are more images to process."""
        return self.current_idx < len(self.dataframe) - 1

