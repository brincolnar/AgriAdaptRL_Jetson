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
        self.fc1 = nn.Linear(4, 64)  # Input layer
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

    print('device: ')
    print(device)
    
    # Original network
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Target network
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  # Set the target network to evaluation mode

    loss_fn = nn.SmoothL1Loss() #nn.MSELoss() # nn.MSELoss(), nn.SmoothL1Loss() (other losses)

    # Load the dataset
    features_file = pd.read_csv('./features.csv')
    performance_file = pd.read_csv('./performance_results.csv')

    features = pd.merge(features_file, performance_file, left_on='Filename', right_on='Filename')

    performance_columns = ['0%', '25%', '50%', '75%']
    features['best_network'] = features[performance_columns].idxmax(axis=1)

    train_filenames = os.listdir('./data/ordered_train_test/train/images/')
    test_filenames = os.listdir('./data/ordered_train_test/test/images')

    # Train-test split
    train_df = features[features['Filename'].isin(train_filenames)]
    test_df = features[features['Filename'].isin(test_filenames)]

    combined_df = pd.concat([train_df, test_df])

    # selected_features = [
    # "Mean Brightness",
    # "Std Brightness",
    # "Max Brightness",
    # "Min Brightness",
    # "Hue Hist Feature 1",
    # "Hue Hist Feature 2",
    # "Hue Std",
    # "Contrast",
    # "Mean Saturation",
    # "Std Saturation",
    # "Max Saturation",
    # "Min Saturation",
    # "SIFT Features",
    # "Texture Contrast",
    # "Texture Dissimilarity",
    # "Texture Homogeneity",
    # "Texture Energy",
    # "Texture Correlation",
    # "Texture ASM",
    # "Excess Green Index",
    # "Excess Red Index",
    # "CIVE",
    # "ExG-ExR Ratio",
    # "CIVE Ratio"]


    # selected_features = [
    #     "Texture Correlation",
    #     "Texture Dissimilarity",
    #     "Texture Contrast",
    #     "Texture Homogeneity",
    # ]


    selected_features = [
        'Hue Hist Feature 1', 'Max Brightness', 'Hue Hist Feature 2', 'Excess Red Index'
    ]


    verbose = False 

    results = []

    features_str = ", ".join(selected_features)

    print("========================================================")
    print(f"Evaluating for feature/s: {features_str}")
    print("========================================================")

    # Calculate normalization parameters using only the training data
    feature_means = [train_df[feature].mean() for feature in selected_features]
    feature_stds = [train_df[feature].std() for feature in selected_features]

    test_df = features[features['Filename'].isin(test_filenames)]
    
    train_env = NetworkSelectionEnv(dataframe=train_df, 
        features=selected_features, 
        feature_means=feature_means, 
        feature_stds=feature_stds, 
        performance_factor=PERFORMANCE_FACTOR, 
        verbose=False
    )

    replay_buffer = ReplayBuffer(capacity=BATCH_SIZE)  

    # Train
    for i in range(NUM_ITERATIONS):
        epsilon = EPSILON_START
        losses = []

        print("========================================================")
        print(f"Iteration {i} starting...")
        print("========================================================")

        state = train_env.reset()  # Reset the environment
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        
        for t in tqdm(range(len(train_df)), desc=f'Iteration {i}'):
            # Select action
            if random.random() > epsilon:
                with torch.no_grad():
                    action = model(state).argmax().item()
            else:
                action = train_env.sample_action()

            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** t))  # Decay epsilon

            # Execute action
            next_state, reward, done = train_env.get_next_state_reward(action)
            next_state = torch.FloatTensor(next_state).to(device).unsqueeze(0)

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

                # print(f'Loss at step {t}: {loss.item()}')

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state

        plt.figure(figsize=(10,5))
        plt.plot(losses, label='Loss')
        plt.xlabel('Training Steps')    
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"./loss.png", bbox_inches='tight')

        actions_taken = []

        # Evaluate the model
        total_test_reward, accuracy, average_weight, average_iou, correct_predictions, total_predictions, actions_taken = evaluate_model(
            model=model,
            optimizer=optimizer, 
            features=selected_features, 
            test_data=test_df, 
            feature_means=feature_means,
            feature_stds=feature_stds,
            performance_factor=PERFORMANCE_FACTOR,
        )

        print(f"Average accuracy: {accuracy:.2f}")
        print(f"Average IoU: {average_iou:.2f}")
        print(f"Average weight: {average_weight:.2f}")
        print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")
        print(f"Features used: {', '.join(selected_features)}")

        # Redesign this function for multiple features
        # plot_test_split(i, accuracy)

    print("----------------------------------------------------------")
