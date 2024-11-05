import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm 
from Segment import Segment  # Import the Segment class
from time import time
import cv2
from spectral_features import SpectralFeatures
from texture_features import TextureFeatures
from vegetation_features import VegetationIndices

'''
Class for training, testing contextual bandit using linear 
'''
class ContextBandit:
    def __init__(self, n_actions, n_features, image_dir, label_dir, model_paths, performance_factor, epochs, learning_rate=0.01, epsilon=0.10):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.window = 100000
        self.epsilon = epsilon
        self.weights = np.random.normal(size=(n_actions, n_features))
        self.scaler = StandardScaler()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.PERFORMANCE_FACTOR = performance_factor
        self.inference_dim = (256, 256)

        # self.feature_columns = [
        #     'Mean Brightness', 'Hue Hist Feature 1', 'Mean Saturation',
        #     'Std Brightness', 'Max Brightness', 'Min Brightness',
        #     'Hue Hist Feature 2', 'Hue Std', 'Contrast', 'Std Saturation',
        #     'Max Saturation', 'Min Saturation', 'Texture Contrast',
        #     'Texture Dissimilarity', 'Texture Homogeneity', 'Texture Energy',
        #     'Texture Correlation', 'Texture ASM', 'Excess Green Index',
        #     'Excess Red Index', 'CIVE', 'ExG-ExR Ratio', 'CIVE Ratio'
        # ]

        # self.feature_columns = [
        #     'Texture Correlation', 'Texture Dissimilarity', 'Texture Contrast',
        #     'Texture Homogeneity'
        # ]

        self.feature_columns = [
            'Hue Hist Feature 1', 'Max Brightness', 'Hue Hist Feature 2', 'Excess Red Index'
        ]

        self.current_image_index = 0
        self.verbose = False
        self.rewards = []
        self.prediction_errors = []
        self.action_to_network = {0: "0%", 1: "25%", 2: "50%", 3: "75%"}
        self.action_to_network_inverse = {v: k for k, v in self.action_to_network.items()}
        
        # Initialize Segment models for each network level (0%, 25%, 50%, 75% pruned)
        self.segments = {
            i: Segment(model_path=model_paths[i], image_dir=image_dir, label_dir=label_dir, resolution=self.inference_dim, device="cuda")
            for i in range(n_actions)
        }

        # Prepare image list for training
        self.image_list = sorted(os.listdir(image_dir))


    def plot_metrics(self):

        rewards_series = pd.Series(self.rewards)
        smoothed_rewards = rewards_series.rolling(window=self.window).mean()

        plt.figure(figsize=(6, 4))        
        plt.plot(smoothed_rewards, label='Rewards')
        plt.title('Smoothed Rewards over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('./smoothed_rewards.png')
        plt.close()

        prediction_series = pd.Series(self.prediction_errors)
        smoothed_predictions = prediction_series.rolling(window=self.window).mean()

        plt.figure(figsize=(6, 4))
        plt.plot(self.prediction_errors, label='Prediction Errors', color='red')
        plt.title('Prediction Errors over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Prediction Error')
        plt.legend()
        plt.savefig('./smoothed_prediction_errors.png')
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
        plt.savefig('./epsilongreedy_iou.png')
        plt.close()

    def compute_image_features(self, image_path):        
        image_path = os.path.join(self.image_dir, image_path)

        print('image_path')
        print(image_path)

        image = cv2.imread(image_path)
        
        # Spectral features
        spectral = SpectralFeatures(image)
        mean_brightness, std_brightness, max_brightness, min_brightness = spectral.compute_brightness()
        hue_hist1, hue_contrast, hue_std = spectral.compute_hue_histogram()
        mean_saturation, std_saturation, max_saturation, min_saturation = spectral.compute_saturation()

        # Texture features
        texture = TextureFeatures(image)
        glcm = texture.compute_glcm()
        texture_contrast = texture.contrast_feature(glcm)
        texture_dissimilarity = texture.dissimilarity_feature(glcm)
        texture_homogeneity = texture.homogeneity_feature(glcm)

        # Vegetation indices
        veg = VegetationIndices(image)
        excess_green = veg.excess_green_index()
        excess_red = veg.excess_red_index()
        cive = veg.colour_index_vegetation_extraction()
        exg_exr_ratio = veg.excess_green_excess_red_index(excess_green, excess_red)

        # Choose only the necessary features based on self.feature_columns
        features = [
            hue_hist1, max_brightness, hue_contrast, excess_red.mean()
        ]
        
        return np.array(features, dtype=float)


    def load_image_features(self, image_path):
        features = self.compute_image_features(image_path)
        return features

    def predict_rewards(self, context):
        # context = self.scaler.transform([context])[0]
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
        context = self.scaler.transform([context])[0]
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

    def calculate_test_reward(self, row, selected_network):
        selected_network_name = self.action_to_network[selected_network]
        selected_network_performance = row[selected_network_name]
        network_to_normalized_weight = {"0%": 0.00, "25%": 0.25, "50%": 0.50, "75%": 0.75}
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network_name]
        return reward
    
    def train(self, epochs):
        # Training
        total_actions_taken = np.zeros(self.n_actions)
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        true_actions = []
        total_reward = 0
        ious = []
        filename_iou_list = []  

        for epoch in range(epochs):
            for image_name in tqdm(self.image_list, desc="Training Progress"):
                image_start_time = time()   

                print('image_name')
                print(image_name)

                features = self.load_image_features(image_name)
                print('features: ')
                print(features)
                action = bandit.select_action(features)

                reward, iou = self.calculate_reward(action, image_name)

                total_iou += iou
                ious.append(iou)

                network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}

                filename = row['Filename']
                filename_iou_list.append({'Filename': filename, 'Epsilon': selected_network_iou, 'EpsilonWeight': network_to_weight[selected_network_name] })     
            
                true_action = self.action_to_network_inverse[row["best_network"]]
                true_actions.append(true_action) 

                if action == true_action:
                    correct_selections += 1

                total_weight += network_to_weight[selected_network_name]

                reward = bandit.calculate_reward(action)
                next_action = bandit.fit(features, action, reward)
                bandit.current_image_index += 1  
            
            average_iou = total_iou / len(self.dataset) 
            average_weight = total_weight / len(self.dataset) 
            accuracy = correct_selections / len(self.dataset)

            print(f"Average IoU: {average_iou:.2f}")
            print(f"Average Weight: {average_weight:.2f}")
            print(f"Accuracy: {accuracy * 100:.2f}%")

        self.plot_iou(ious)
        self.plot_metrics()

        results_df = pd.DataFrame(filename_iou_list)
        results_df.to_csv('epsilonbandit_results.csv', index=False)
        print("Results saved to 'epsilonbandit_results.csv'")

    def test(self):
        print("Starting testing phase...")
        total_actions_taken = np.zeros(self.n_actions)
        correct_selections = 0
        total_iou = 0  
        total_weight = 0 
        true_actions = []
        total_reward = 0

        for index, row in self.test_df.iterrows():
            features = row[self.feature_columns].values.astype(float)
            features = self.scaler.transform([features])[0]  
            predicted_rewards = np.dot(self.weights, features)  
            selected_action = np.argmax(predicted_rewards)  

            true_action = self.action_to_network_inverse[row["best_network"]]
            true_actions.append(true_action) 

            reward = self.calculate_test_reward(row, selected_action) 
            total_reward += reward
            total_actions_taken[selected_action] += 1

            if selected_action == true_action:
                correct_selections += 1

            selected_network_name = self.action_to_network[selected_action]
            selected_network_iou = row[selected_network_name]
            total_iou += selected_network_iou
            
            network_to_weight = {"0%": 100, "25%": 75, "50%": 50, "75%": 25}
            total_weight += network_to_weight[selected_network_name]

        total_tests = len(self.test_df)
        average_iou = total_iou / total_tests if total_tests > 0 else 0
        average_weight = total_weight / total_tests if total_tests > 0 else 0
        accuracy = correct_selections / total_tests if total_tests > 0 else 0
        
        print("Testing completed.")
        print(f"Actions Taken: {total_actions_taken}")
        print(f"True Actions: {true_actions}")
        print(f"Average IoU: {average_iou:.2f}")
        print(f"Average Weight: {average_weight:.2f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Total Reward: {total_reward}")

        return total_actions_taken, accuracy


if __name__ == "__main__":
    image_dir = "./data/ordered_train_test/all/images"
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
        n_features=4, 
        image_dir=image_dir, 
        label_dir=label_dir,
        model_paths=model_paths,
        performance_factor=performance_factor, 
        epochs=epochs)

    # Training
    bandit.train(epochs)

    # Testing
    # bandit.test()