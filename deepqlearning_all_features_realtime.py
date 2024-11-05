import re as re
import csv
import torch
import random
import os 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from NetworkSelectionEnv import NetworkSelectionEnv
from ReplayBuffer import ReplayBuffer
from utils import evaluate_model, plot_test_split
# from hyperparams import STEPS_PER_EPISODE
from hyperparams import EPISODES, EPSILON, EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, LEARNING_RATE, PERFORMANCE_FACTOR, NUM_ITERATIONS, BATCH_SIZE, TARGET_UPDATE

network_performance_columns = [
    '0%', '25%', '50%', '75%'
]

# Create a mapping from action index to network name
action_to_network_mapping = {index: name for index, name in enumerate(network_performance_columns)}

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(23, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32) # Hidden layer

        # add hidden layers
        self.fc3 = nn.Linear(32, 4)  # Output layer: Adjust the number of outputs to match your networks

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

if __name__ == "__main__":

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=================================================')
    print(f'DEVICE: {device}')
    
    # Original network
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Target network
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  # Set the target network to evaluation mode

    loss_fn = nn.SmoothL1Loss() #nn.MSELoss() # nn.MSELoss(), nn.SmoothL1Loss() (other losses)

    selected_features = [
    "Mean Brightness",
    "Std Brightness",
    "Max Brightness",
    "Min Brightness",
    "Hue Hist Feature 1",
    "Hue Hist Feature 2",
    "Hue Std",
    "Contrast",
    "Mean Saturation",
    "Std Saturation",
    "Max Saturation",
    "Min Saturation",
    "SIFT Features",
    "Texture Contrast",
    "Texture Dissimilarity",
    "Texture Homogeneity",
    "Texture Energy",
    "Texture Correlation",
    "Texture ASM",
    "Excess Green Index",
    "Excess Red Index",
    "CIVE",
    "ExG-ExR Ratio",
    "CIVE Ratio"]


    verbose = False 

    results = []

    features_str = ", ".join(selected_features)

    print("========================================================")
    print(f"Evaluating for feature/s: {features_str}")
    print("========================================================")
    
    # Initialize environment with model paths
    env = NetworkSelectionEnvImages(
        image_folder='./data/ordered_train_test/test/images/',
        label_folder='./data/ordered_train_test/test/labels/', 
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
        # epsilon = EPSILON_START
        epsilon = EPSILON
        losses = []

        print("========================================================")
        print(f"Iteration {i} starting...")
        print("========================================================")

        state = train_env.reset()  # Reset the environment
        state = torch.FloatTensor(state).to(device).unsqueeze(0)

        filenames = []
        ious = []
        weights = []
        battery_levels = []
        image_times = []
        
        for t in tqdm(range(len(train_df)), desc=f'Iteration {i}'):
            image_start_time = time()

            # Select action
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).argmax().item()
            else:
                action = train_env.sample_action()

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

        results_df.to_csv('deepqlearning_features_results.csv', index=False)
        print("Results have been saved to 'deepqlearning_features_results.csv'.")

        average_weight = sum(weights) / len(weights)
        average_iou = sum(ious) / len(ious) 
        last_battery_level = battery_levels[-1]
        avg_image_time = sum(image_times) / len(image_times)

        print(f"Average Weight: {average_weight}")
        print(f"Average IoU: {average_iou}")
        print(f"Last Battery Level: {last_battery_level}")
        print(f"Average Time per Image: {avg_image_time:.4f} seconds")


        # plt.figure(figsize=(10,5))
        # plt.plot(losses, label='Loss')
        # plt.xlabel('Training Steps')    
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig(f"./loss.png", bbox_inches='tight')

        # actions_taken = []

        # Evaluate the model
        # total_test_reward, accuracy, average_weight, average_iou, correct_predictions, total_predictions, actions_taken = evaluate_model(
        #     model=model,
        #     optimizer=optimizer, 
        #     features=selected_features, 
        #     test_data=test_df, 
        #     feature_means=feature_means,
        #     feature_stds=feature_stds,
        #     performance_factor=PERFORMANCE_FACTOR,
        # )

        # print(f"Average accuracy: {accuracy:.2f}")
        # print(f"Average IoU: {average_iou:.2f}")
        # print(f"Average weight: {average_weight:.2f}")
        # print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")
        # print(f"Features used: {', '.join(selected_features)}")

        # Redesign this function for multiple features
        # plot_test_split(i, accuracy)

    print("----------------------------------------------------------")
