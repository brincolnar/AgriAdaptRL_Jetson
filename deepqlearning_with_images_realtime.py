import csv
import torch
import random
import os 
import torch
import math
import torch.nn.functional as F
import re as re
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split
from NetworkSelectionEnvImages import NetworkSelectionEnvImages
from ReplayBuffer import ReplayBuffer
from utils_images import evaluate_model, plot_test_split, moving_average, evaluate_test
from hyperparams import EPISODES, EPSILON, EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, LEARNING_RATE, PERFORMANCE_FACTOR, NUM_ITERATIONS, BATCH_SIZE, TARGET_UPDATE

network_performance_columns = [
    '0%', '25%', '50%', '75%'
]

# Create a mapping from action index to network name
action_to_network_mapping = {index: name for index, name in enumerate(network_performance_columns)}

network_to_weight_mapping = [100, 75, 50, 25]

class ConvDQN(nn.Module):

    # Channels: 
    # 1. R chanel of image, 
    # 2. G chanel of image, 
    # 3. B chanel of image, 
    # 4. normalized battery level, 
    # 5. normalized images left     
    def __init__(self, input_shape=(5, 84, 84)):  
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the output size of the last convolutional layer
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 4)  # Assuming 4 actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=================================================')
    print(f'DEVICE: {device}')

    # Original network
    model = ConvDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Target network
    target_model = ConvDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  # Set the target network to evaluation mode

    loss_fn = nn.SmoothL1Loss() # nn.MSELoss(), nn.SmoothL1Loss() (other losses)

    verbose = True 
    results = []
    
    # Initialize environment with model paths
    env = NetworkSelectionEnvImages(
        image_folder='./data/ordered_train_test/all/images/',
        label_folder='./data/ordered_train_test/all/labels/', 
        performance_factor=PERFORMANCE_FACTOR,
        device=device,
        model_paths=[
            "./garage/unet_512_pruned_00_iterative_1.pt",          # Path to unpruned model
            "./garage/unet_512_pruned_025_iterative_1.pt",         # Path to 25% pruned model
            "./garage/unet_512_pruned_05_iterative_1.pt",          # Path to 50% pruned model
            "./garage/unet_512_pruned_075_iterative_1.pt"          # Path to 75% pruned model
        ],
        resize_dim=(84, 84),  # For DQN input
        inference_dim=(512, 512)  # For full-size inference
    )


    replay_buffer = ReplayBuffer(capacity=BATCH_SIZE)

    # Measure total training time
    total_start_time = time()

    # Train
    for i in range(NUM_ITERATIONS):
        iteration_start_time = time()
        epsilon = EPSILON_START
        losses = []

        print("========================================================")
        print(f"Iteration {i} starting...")
        print("========================================================")

        state = env.reset()  # Reset the environment
        state = state.to(device).unsqueeze(0)

        filenames = []
        ious = []
        weights = []
        battery_levels = []
        image_times = []

        for t in tqdm(range(len(env.image_filenames)), desc=f'Iteration {i}'):
            image_start_time = time()
            
            # Select action
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).argmax().item()
            else:
                action = env.sample_action()
            
            # epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** t))  # Decay epsilon
            epsilon = EPSILON

            # Execute action
            next_state, reward, iou, done = env.get_next_state_reward(action)
            next_state = next_state.unsqueeze(0) 

            current_filename = env.image_filenames[env.current_idx]
            filenames.append(current_filename)
            ious.append(iou)
            weights.append(network_to_weight_mapping[action])
            battery_levels.append(env.battery)

            print(f"Current filename: {current_filename}, IoU: {iou}")

            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) >= BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state).float()
                batch_action = torch.tensor(batch_action, device=device).long()  # Actions are usually expected to be long
                batch_reward = torch.tensor(batch_reward, device=device).float()
                batch_next_state = torch.cat(batch_next_state).float()
                batch_done = torch.tensor(batch_done, device=device).float()

                # Compute Q-values and loss 
                current_q_values = model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(batch_next_state).max(1)[0].detach()  # Detach from graph to avoid backpropagation to target model
                expected_q_values = batch_reward + GAMMA * next_q_values

                # Update target network
                if t % TARGET_UPDATE == 0:
                    target_model.load_state_dict(model.state_dict())

                loss = loss_fn(current_q_values, expected_q_values)
                losses.append(loss.item())

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

            image_end_time = time()
            image_times.append(image_end_time - image_start_time)
            print(f"Time taken for {current_filename}: {image_end_time - image_start_time:.4f} seconds")

        iteration_end_time = time()
        print(f"Iteration {i} completed in {iteration_end_time - iteration_start_time:.2f} seconds.")

        results_df = pd.DataFrame({
            'Filename': filenames,
            'DQN': ious,
            'DQNWeight': weights,
            'ImageTime': image_times
        })

        results_df.to_csv('deepqlearning_results.csv', index=False)
        print("Results have been saved to 'deepqlearning_results.csv'.")

        average_weight = sum(weights) / len(weights)
        average_iou = sum(ious) / len(ious) 
        last_battery_level = battery_levels[-1]
        avg_image_time = sum(image_times) / len(image_times)

        print(f"Average Weight: {average_weight}")
        print(f"Average IoU: {average_iou}")
        print(f"Last Battery Level: {last_battery_level}")
        print(f"Average Time per Image: {avg_image_time:.4f} seconds")

        # evaluate_test(ious, weights, battery_levels)

        total_images = len(weights)
        third = total_images // 3

        first_third_avg_weight = sum(weights[:third]) / third
        second_third_avg_weight = sum(weights[third:2*third]) / third
        third_third_avg_weight = sum(weights[2*third:]) / (total_images - 2*third)

        moving_average_ious = moving_average(ious, window_size=5)
        moving_average_weights = moving_average(weights, window_size=5)
        moving_average_battery_levels = moving_average(battery_levels, window_size=5)

        plt.figure(figsize=(15, 15))

        # Plot for IoUs
        plt.subplot(3, 1, 1)  # This means 3 rows, 1 column, and this plot is the 1st
        plt.plot(moving_average_ious, label='IoU', color='blue')
        plt.xlabel('Image No.')
        plt.ylabel('IoU')
        plt.title('IoU Over Images')
        plt.legend()

        # Plot for Weights
        plt.subplot(3, 1, 2)  # This means 3 rows, 1 column, and this plot is the 2nd
        plt.plot(moving_average_weights, label='Weight', color='green')
        plt.xlabel('Image No.')
        plt.ylabel('Weight')
        plt.title('Weight Over Images')
        plt.legend()

        plt.text(x=max(len(weights)*0.1, 10), y=max(weights)*0.8, s=f'1st Third Avg Weight: {first_third_avg_weight:.2f}')
        plt.text(x=max(len(weights)*0.45, 10), y=max(weights)*0.8, s=f'2nd Third Avg Weight: {second_third_avg_weight:.2f}')
        plt.text(x=max(len(weights)*0.8, 10), y=max(weights)*0.8, s=f'3rd Third Avg Weight: {third_third_avg_weight:.2f}')

        # Plot for Battery Levels
        plt.subplot(3, 1, 3)  # This means 3 rows, 1 column, and this plot is the 3rd
        plt.plot(moving_average_battery_levels, label='Battery Level', color='red')
        plt.xlabel('Image No.')
        plt.ylabel('Battery Level')
        plt.title('Battery Level Over Images')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"./results.png", bbox_inches='tight')

