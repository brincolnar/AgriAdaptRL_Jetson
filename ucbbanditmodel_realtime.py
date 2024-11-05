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

class ContextBanditUCB:
    def __init__(self, n_actions, performance_factor, n_features, image_dir, label_dir, model_paths, c=1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.c = c  # Exploration parameter for UCB
        self.weights = np.random.normal(size=(n_actions, n_features))
        self.action_counts = np.zeros(n_actions)  # Count of selections for each action
        self.sum_rewards = np.zeros(n_actions)    # Sum of rewards for each action
        self.average_rewards = np.zeros(n_actions)
        self.total_iterations = 0
        self.scaler = StandardScaler()
        self.PERFORMANCE_FACTOR = performance_factor
        self.learning_rate = 0.01

        # Load and prepare the dataset
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
        self.prediction_errors = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}
        
        # Initialize Segment models for each network level (0%, 25%, 50%, 75% pruned)
        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_dir, label_dir=label_dir, resolution=(512, 512), device="cuda")
            for i in range(n_actions)
        }
        
        # For plotting
        self.window = 10000
        self.ucb_values = {i: [] for i in range(self.n_actions)}

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

    def plot_ucb_values(self):
        plt.figure(figsize=(10, 6))
        for action in range(self.n_actions):
            plt.plot(self.ucb_values[action], label=f'Action {action} ({self.action_to_network[action]})')
        plt.title('UCB Values over Time')
        plt.xlabel('Steps')
        plt.ylabel('UCB Value')
        plt.legend()
        plt.savefig('./ucb_values_over_time.png')
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
        plt.savefig('./ucb_iou.png')
        plt.close()

    def load_image_features(self):
        if self.current_image_index < len(self.dataset):
            row = self.dataset.iloc[self.current_image_index]
            features = row[self.feature_columns].values.astype(float)
            return features, row
        else:
            return None, None

    def select_action(self, context):
        scaled_context = self.scaler.transform([context])
        
        # Calculate UCB values for each action
        ucb_values = self.average_rewards + self.c * np.sqrt(np.log(self.total_iterations + 1) / (self.action_counts + 1))

        # Store UCB values for plotting
        for action in range(self.n_actions):
            self.ucb_values[action].append(ucb_values[action])        
        
        action = np.argmax(ucb_values)
        return action

    def calculate_reward(self, selected_network, image_name):
        # Use Segment to infer and calculate IoU score for the selected network
        results, _, _, _ = self.segments[selected_network].infer(image_name)
        selected_network_iou = results['test/iou/weeds']
        
        # Calculate reward based on IoU and performance factor
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_iou * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[self.action_to_network[selected_network]]
        
        return reward, selected_network_iou

    def update(self, action, reward, context):
        self.action_counts[action] += 1
        self.sum_rewards[action] += reward
        self.average_rewards[action] = self.sum_rewards[action] / self.action_counts[action]
        self.total_iterations += 1

        # Update weights using linear regression update rule
        prediction = np.dot(self.weights[action], context)
        error = reward - prediction
        self.weights[action] += self.learning_rate * error * context

    def train(self, epochs):
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
                    self.current_image_index = 0
                    break

                action = self.select_action(features)

                reward, iou = self.calculate_reward(action, image_name)

                print(f'image_name: {image_name}, iou: {iou}')
                
                total_iou += iou
                ious.append(iou)

                network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
                total_weight += network_to_weight[self.action_to_network[action]]

                self.rewards.append(reward)
                self.update(action, reward, features)
                self.current_image_index += 1  

                image_end_time = time()  # End timing for each image
                image_times.append(image_end_time - image_start_time)
                print(f"Time taken for {image_name}: {image_end_time - image_start_time:.4f} seconds")


            # Calculate and print average metrics for this epoch
            average_iou = total_iou / len(self.image_list)
            average_weight = total_weight / len(self.image_list)
            print(f"Epoch {epoch+1}: Average IoU: {average_iou:.2f}, Average Weight: {average_weight:.2f}")

        end_time = time()
        avg_image_time = sum(image_times) / len(image_times)
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print(f"Average time per image: {avg_image_time:.4f} seconds.")


        self.plot_iou(ious)
        self.plot_metrics()
        self.plot_ucb_values()

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

    bandit = ContextBanditUCB(
        n_actions=4,
        performance_factor=performance_factor,
        n_features=23,
        image_dir=image_dir,
        label_dir=label_dir,
        model_paths=model_paths,
        c=1
    )

    # Train the bandit
    bandit.train(epochs)
