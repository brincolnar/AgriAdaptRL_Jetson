import csv
import os
import torch
import random
import re as re
import torch.nn as nn
import matplotlib.pyplot as plt
from NetworkSelectionEnv import NetworkSelectionEnv
from hyperparams import EPSILON, GAMMA

def exponential_moving_average(values, alpha=0.1):
    """ Calculate the exponential moving average of a sequence of numbers. """
    average = 0
    for value in values:
        average = alpha * value + (1 - alpha) * average
    return average

# Function to ensure the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_model(model, optimizer, features, feature_means, feature_stds, test_data, performance_factor):

    network_performance_columns = [
        '0%', '25%', '50%', '75%'
    ]

    # Put the model in evaluation mode
    model.eval()

    test_env = NetworkSelectionEnv(dataframe=test_data, 
    features=features, 
    feature_means=feature_means, 
    feature_stds=feature_stds,
    performance_factor=performance_factor, 
    verbose=False)

    # Track total reward and actions taken
    total_reward = 0
    correct_predictions = 0
    total_predictions = 0

    state = test_env.reset() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.FloatTensor(state).to(device)
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
        next_state = torch.FloatTensor(next_state).to(device)
        next_state = next_state.unsqueeze(0) # add batch dimension
        
        target = reward + GAMMA * model(next_state).max().item() * (not done) 
        target = torch.tensor([target], device=device, dtype=torch.float32) 

        if verbose:
            print(f"next_state: {next_state}")
            print(f"action: {action}")

        # Get the best network and selected network performance
        selected_network = network_performance_columns[action]

        total_reward += reward
        
        total_iou += test_env.dataframe.iloc[test_env.current_idx][selected_network]

        actions_taken.append((test_env.dataframe.iloc[test_env.current_idx]['Filename'], action))

        # Add weight
        total_weight += network_to_weight[selected_network]

        best_network = test_env.dataframe.iloc[test_env.current_idx]['best_network']

        if verbose:
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

    # name = f"{'_'.join(features)}"
    # with open(f'./results/predictions_result_{name}.csv', 'w', newline='') as file:
    #     fieldnames = ['Filename', 'Predicted Network', 'Best network']
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(results)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    average_weight = total_weight / total_predictions
    average_iou = total_iou / total_predictions

    return total_reward, accuracy, average_weight, average_iou, correct_predictions, total_predictions, actions_taken

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