import csv
import os
import torch
import random
import math
import re as re
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from NetworkSelectionEnvImages import NetworkSelectionEnvImages
from hyperparams import EPSILON, GAMMA

def plot_average_weight_trend(all_test_weights):
    max_length = max(len(weights) for weights in all_test_weights)

    # Initialize a numpy array to store cumulative sums for each position
    weight_sums = np.zeros(max_length)
    weight_counts = np.zeros(max_length)

    # Accumulate weights for each position across all test files
    for weights in all_test_weights:
        for idx, weight in enumerate(weights):
            weight_sums[idx] += weight
            weight_counts[idx] += 1

    # Calculate the average weight for each position
    average_weights = weight_sums / weight_counts

    # Plot the average weights
    plt.figure(figsize=(12, 6))
    plt.plot(average_weights, label='Average Weight', color='green')
    plt.xlabel('Image Index')
    plt.ylabel('Average Weight')
    plt.title('Average Weight Movement Across the Entire Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig('weight_movement.png')

def moving_average(values, window_size):
    """ Calculate the simple moving average of a sequence of numbers given a specific window size. """
    smas = []
    # Calculate SMA for each point after enough values are accumulated to form the first window
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        smas.append(sum(window) / window_size)
    return smas


# Function to ensure the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_model(model, optimizer, test_data, performance_factor):

    network_performance_columns = [
        '0%', '25%', '50%', '75%'
    ]

    # Put the model in evaluation mode
    model.eval()

    # test_env = NetworkSelectionEnvImages(dataframe=test_data, 
    # image_folder='./data/ordered_train_test/test/images/', 
    # verbose=False,
    # device='cuda',
    # performance_factor=performance_factor)

    test_env = NetworkSelectionEnvImages(dataframe=test_data, 
    image_folder='./data/ordered_train_test/all/images/', 
    performance_factor=performance_factor, 
    device='cuda', 
    verbose=False)


    # Track total reward and actions taken
    total_reward = 0
    correct_predictions = 0
    total_predictions = 0

    state = test_env.reset() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = state.unsqueeze(0) 
    loss_fn = nn.SmoothL1Loss()

    results = []

    verbose = False
    network_to_weight = {
        "0%": 100,
        "25%": 75,
        "50%": 50,
        "75%": 25
    }

    total_weight = 0
    total_iou = 0

    actions_taken = []
    ious = []
    weights = []
    battery_levels = []

    # Loop over the test data
    while test_env.has_next():
        
        # Select action
        if random.random() > EPSILON:
            with torch.no_grad():
                action = model(state).argmax().item()
        else:
            action = test_env.sample_action()

        q_values = None
        action = None

        q_values = model(state)
        action = model(state).argmax().item()

        prediction = q_values[0][action].unsqueeze(0)

        if verbose:
            print(f"q_values: {q_values}")
            print(f"action: {action}")

        next_state, reward, done = test_env.get_next_state_reward(action)
        next_state = next_state.unsqueeze(0) # add batch dimension
        
        target = reward + GAMMA * model(next_state).max().item() * (not done) 
        target = torch.tensor([target], device=device, dtype=torch.float32) 

        if verbose:
            print(f"next_state: {next_state}")
            print(f"action: {action}")

        # Get the best network and selected network performance
        selected_network = network_performance_columns[action]

        total_reward += reward
        
        #  Add IoU
        total_iou += test_env.dataframe.iloc[test_env.current_idx][selected_network]

        ious.append(test_env.dataframe.iloc[test_env.current_idx][selected_network])

        # Add weight
        total_weight += network_to_weight[selected_network]

        weights.append(network_to_weight[selected_network])

        # Add battery level 
        battery_levels.append(test_env.battery)


        best_network = test_env.dataframe.iloc[test_env.current_idx]['best_network']


        if verbose:
            print('Verbose')
            print(verbose)
            
            print('Filename: ')
            print(test_env.dataframe.iloc[test_env.current_idx]['Filename'])

            print('Selected network: ')
            print(selected_network)

            print('Best network: ')
            print(best_network)

        results.append({
            "Filename": test_env.dataframe.iloc[test_env.current_idx]['Filename'],
            "Predicted Network": selected_network,
            "Best network": best_network
        })

        if selected_network == best_network:
            correct_predictions += 1

        total_predictions += 1                
        
        loss = loss_fn(prediction, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # IMPORTANT: Update the state with the next state for the next iteration
        state = next_state

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    average_weight = total_weight / total_predictions
    average_iou = total_iou / total_predictions

    return total_reward, accuracy, average_weight, average_iou, correct_predictions, total_predictions, actions_taken, ious, weights, battery_levels

def plot_test_split(num_repeat, accuracy):
    plt.figure()

    # Extract the train and test data for the feature and best network performances
    x_train = train_df[feature].tolist()
    x_test = [test_df[test_df['Filename'] == action[0]][feature].values[0] for action in actions_taken]

    best_networks = train_df['best_network'].tolist()
    best_networks_ = test_df['best_network'].tolist()

    print('best_networks_')
    print(best_networks_)
    
    y_train = [train_df.iloc[i][best_network] for i, best_network in enumerate(best_networks)]
    y_test = [test_df[test_df['Filename'] == action[0]][network_performance_columns[action[1]]].values[0] for action in actions_taken]
    truth = [test_df.iloc[i][best_network] for i, best_network in enumerate(best_networks_)]

    # Plot training and testing datasets
    plt.scatter(x_train, y_train, marker='o', color='blue', label='Train')
    plt.scatter(x_test, y_test, marker='h', color='red', label='Test')
    plt.scatter(x_test, truth, marker='x', color='yellow', label='Truth')

    # Adding legend
    plt.legend()

    # Adding title
    plt.title(f'Results for {feature}, Accuracy {round(accuracy, 2)}')

    # Adding labels
    plt.xlabel(feature)
    plt.ylabel('Performance')

    # Save the plot
    plt.savefig(f"./plots/test_plots/{feature.replace(' ', '')}_{num_repeat}_repeat.png")

def evaluate_test(ious, weights, battery_levels):
    test_size = math.floor(len(ious) * 0.2)
    print(f'test_size: {test_size}')
    last_50_ious = ious[-test_size:]  
    last_50_weights = weights[-test_size:]
    last_50_battery_levels = battery_levels[-test_size:]  

    average_last_50_ious = np.mean(last_50_ious)
    average_last_50_weights = np.mean(last_50_weights)
    average_last_50_battery_levels = np.mean(last_50_battery_levels)

    print("----------------------------------------------------------")
    print(f"Average IoU for the last {test_size} images: {average_last_50_ious:.2f}")
    print(f"Average Weight for the last {test_size} images: {average_last_50_weights:.2f}")
    print(f"Average Battery Level for the last {test_size} images: {average_last_50_battery_levels:.2f}")
    print("----------------------------------------------------------")
    