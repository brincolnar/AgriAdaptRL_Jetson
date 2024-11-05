import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from time import time
from torchvision import transforms
from Segment import Segment  # Import the Segment class
from PIL import Image

class ContextBandit:
    def __init__(self, n_actions, performance_factor, image_dir, label_dir, model_paths, epochs, resize_dim=(84, 84), inference_dim=(256, 256), learning_rate=0.01, epsilon=0.10):
        self.n_actions = n_actions
        self.n_features = resize_dim[0] * resize_dim[1] * 3
        self.learning_rate = learning_rate
        self.window = 100000
        self.epsilon = epsilon
        self.weights = np.random.normal(size=(n_actions, self.n_features))
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.PERFORMANCE_FACTOR = performance_factor
        self.current_image_index = 0
        self.rewards = []
        self.prediction_errors = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}

        # Initialize Segment models for each network level (0%, 25%, 50%, 75% pruned)
        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_dir, label_dir=label_dir, resolution=inference_dim, device="cuda")
            for i in range(n_actions)
        }

        # Prepare image list for training
        self.image_list = sorted(os.listdir(image_dir))

        # Transform to downscale to 84x84 for feature extraction
        self.feature_transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor()
        ])

    def load_image_features(self, image_name):
        """
        Load and downscale the image to 84x84 for feature extraction.
        """
        image_path = os.path.join(self.image_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        img = self.feature_transform(img)
        features = img.flatten().numpy()  # Flatten to a single feature vector
        return features

    def predict_rewards(self, context):
        predicted_rewards = np.dot(self.weights, context)
        return predicted_rewards

    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            predicted_rewards = self.predict_rewards(context)
            action = np.argmax(predicted_rewards)
        return action

    def fit(self, context, action, reward):
        prediction = np.dot(self.weights[action], context)
        error = reward - prediction
        self.prediction_errors.append(abs(error))
        self.weights[action] += self.learning_rate * error * context
        self.rewards.append(reward)
        return self.select_action(context)

    def calculate_reward(self, selected_network, image_name):
        # Use Segment to infer and calculate IoU score for the selected network
        results, _, _, _ = self.segments[selected_network].infer(image_name)
        selected_network_iou = results['test/iou/weeds']
        
        # Calculate reward based on IoU and performance factor
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_iou * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[self.action_to_network[selected_network]]
        
        return reward, selected_network_iou

    def train(self, epochs):
        start_time = time()  # Start timing the training phase
        total_iou = 0
        total_weight = 0
        correct_selections = 0
        ious = []
        image_times = []  

        for epoch in range(epochs):
            for image_name in tqdm(self.image_list, desc="Training Progress"):
                image_start_time = time()  
                
                print('image_name')
                print(image_name)

                features = self.load_image_features(image_name)  # Load image features at 84x84
                action = self.select_action(features)

                reward, iou = self.calculate_reward(action, image_name)

                print(f'image_name: {image_name}, iou: {iou}')

                total_iou += iou
                ious.append(iou)

                network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
                total_weight += network_to_weight[self.action_to_network[action]]

                self.fit(features, action, reward)

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

    # Additional methods for evaluation and plotting (not shown here)

if __name__ == "__main__":
    image_dir = "./data/ordered_train_test/all/images"        # Directory containing images for segmentation
    label_dir = "./data/ordered_train_test/all/labels"        # Directory containing ground truth labels
    model_paths = [
        "./garage/unet_512_pruned_00_iterative_1.pt",          # Path to unpruned model
        "./garage/unet_512_pruned_025_iterative_1.pt",         # Path to 25% pruned model
        "./garage/unet_512_pruned_05_iterative_1.pt",          # Path to 50% pruned model
        "./garage/unet_512_pruned_075_iterative_1.pt"          # Path to 75% pruned model
    ]
    performance_factor = float(sys.argv[1])
    epochs = int(sys.argv[2])

    bandit = ContextBandit(
        n_actions=4,
        performance_factor=performance_factor,
        image_dir=image_dir,
        label_dir=label_dir,
        model_paths=model_paths,
        epochs=epochs
    )

    # Train the bandit
    bandit.train(epochs)
