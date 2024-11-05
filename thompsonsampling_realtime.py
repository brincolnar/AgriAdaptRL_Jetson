import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Segment import Segment  # Import the Segment class

class ContextBanditThompsonGaussian:
    def __init__(self, n_actions, performance_factor, n_features, image_dir, label_dir, model_paths):
        self.n_actions = n_actions
        self.n_features = n_features
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.PERFORMANCE_FACTOR = performance_factor
        self.scaler = StandardScaler()

        # Initialize mean and variance for Gaussian distributions for each action
        self.means = np.zeros(n_actions)
        self.variances = np.ones(n_actions)

        # Load dataset and features for scaling
        self.features_file = pd.read_csv("./features.csv")
        self.dataset = self.features_file
        self.image_list = sorted(os.listdir(image_dir))  # Get all image names in directory

        self.feature_columns = [
            'Mean Brightness', 'Hue Hist Feature 1', 'Mean Saturation',
            'Std Brightness', 'Max Brightness', 'Min Brightness',
            'Hue Hist Feature 2', 'Hue Std', 'Contrast', 'Std Saturation',
            'Max Saturation', 'Min Saturation', 'Texture Contrast',
            'Texture Dissimilarity', 'Texture Homogeneity', 'Texture Energy',
            'Texture Correlation', 'Texture ASM', 'Excess Green Index',
            'Excess Red Index', 'CIVE', 'ExG-ExR Ratio', 'CIVE Ratio'
        ]

        # Fit the scaler on all feature data at once
        all_train_features = self.dataset[self.feature_columns].values
        self.scaler = StandardScaler().fit(all_train_features)
        self.current_image_index = 0
        self.verbose = False
        self.rewards = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}
        
        # Initialize Segment models for each network level (0%, 25%, 50%, 75% pruned)
        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_dir, label_dir=label_dir, resolution=(512, 512), device="cuda")
            for i in range(n_actions)
        }
        
        # For plotting
        self.window = 10000
        self.ts_samples = {i: [] for i in range(self.n_actions)}  # For tracking sampled values

    def plot_metrics(self):
        rewards_series = pd.Series(self.rewards)
        smoothed_rewards = rewards_series.rolling(window=self.window).mean()

        plt.figure(figsize=(6, 4))        
        plt.plot(smoothed_rewards, label='Rewards')
        plt.title('Smoothed Rewards over Time')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('./smoothed_rewards.png')
        plt.close()

    def plot_ts_samples(self):
        plt.figure(figsize=(10, 6))
        for action in range(self.n_actions):
            plt.plot(self.ts_samples[action], label=f'Action {action} ({self.action_to_network[action]})')
        plt.title('Thompson Sampling Values over Time')
        plt.xlabel('Steps')
        plt.ylabel('Sampled Value')
        plt.legend()
        plt.savefig('./thompson_samples_over_time.png')
        plt.close()
    
    def plot_iou(self, ious):
        ious_series = pd.Series(ious)

        window_size = 5  
        smoothed_ious = ious_series.rolling(window=window_size, min_periods=1).mean()

        # Plot the raw IoU values and the smoothed IoU values
        plt.figure(figsize=(12, 6))
        plt.plot(ious_series, label='IoU', alpha=0.3)
        plt.plot(smoothed_ious, label=f'Smoothed IoU (window={window_size})', linewidth=2)
        plt.title('IoU over Time during Training')
        plt.xlabel('Training Steps')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./thompson_iou.png')
        plt.close()

    def load_image_features(self):
        if self.current_image_index < len(self.dataset):
            row = self.dataset.iloc[self.current_image_index]
            features = row[self.feature_columns].values.astype(float)
            return features, row
        else:
            return None, None

    def select_action(self, context):
        scaled_context = self.scaler.transform([context])[0]
        
        # Thompson Sampling: sample from the Gaussian distribution for each action
        samples = np.maximum(0, np.random.normal(self.means, np.sqrt(self.variances)))

        # Store sampled values for plotting
        for action in range(self.n_actions):
            self.ts_samples[action].append(samples[action])

        action = np.argmax(samples)  # Select the action with the highest sampled value
        return action

    def calculate_reward(self, selected_network, image_name):
        # Use Segment to infer and calculate IoU score for the selected network
        results, _, _, _ = self.segments[selected_network].infer(image_name)
        selected_network_iou = results['test/iou/weeds']
        
        # Calculate reward based on IoU and performance factor
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_iou * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[self.action_to_network[selected_network]]
        
        return reward, selected_network_iou

    def update(self, action, reward):
        # Update the mean and variance using online update rules
        n = len(self.ts_samples[action])  # Number of samples so far

        # Update the mean
        new_mean = (self.means[action] * (n - 1) + reward) / n

        # Update the variance
        new_variance = ((n - 1) * self.variances[action] + (reward - new_mean) * (reward - self.means[action])) / n

        self.means[action] = new_mean
        self.variances[action] = new_variance

    def train(self, epochs):
        start_time = time()  # Start timing the training phase
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        ious = []
        image_times = []  

        # Training
        for epoch in range(epochs):
            for image_name in tqdm(self.image_list, desc="Training Progress"):
                image_start_time = time()  

                features, row = self.load_image_features()
                if features is None:
                    break

                action = self.select_action(features)

                reward, iou = self.calculate_reward(action, image_name)

                print(f'image_name: {image_name}, iou: {iou}')

                total_iou += iou
                ious.append(iou)

                network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
                total_weight += network_to_weight[self.action_to_network[action]]

                self.rewards.append(reward)
                self.update(action, reward)
                self.current_image_index += 1

                image_end_time = time()  # End timing for each image
                image_times.append(image_end_time - image_start_time)
                print(f"Time taken for {image_name}: {image_end_time - image_start_time:.4f} seconds")

            average_iou = total_iou / len(self.image_list)
            average_weight = total_weight / len(self.image_list)
            print(f"Epoch {epoch+1}: Average IoU: {average_iou:.2f}, Average Weight: {average_weight:.2f}")

        end_time = time()
        avg_image_time = sum(image_times) / len(image_times)
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print(f"Average time per image: {avg_image_time:.4f} seconds.")



        self.plot_iou(ious)
        self.plot_metrics()
        self.plot_ts_samples()

if __name__ == "__main__":
    image_dir = "./data/ordered_train_test/test/images"
    label_dir = "./data/ordered_train_test/test/labels"
    model_paths = [
        "./garage/unet_512_pruned_00_iterative_1.pt",
        "./garage/unet_512_pruned_025_iterative_1.pt",
        "./garage/unet_512_pruned_05_iterative_1.pt",
        "./garage/unet_512_pruned_075_iterative_1.pt"
    ]
    performance_factor = float(sys.argv[1])
    epochs = int(sys.argv[2])

    bandit = ContextBanditThompsonGaussian(
        n_actions=4,
        performance_factor=performance_factor,
        n_features=23,
        image_dir=image_dir,
        label_dir=label_dir,
        model_paths=model_paths
    )

    # Train the bandit
    bandit.train(epochs)
