import re 
import random

class NetworkSelectionEnv:
    def __init__(self, image_folder, label_folder, performance_factor, device, model_paths, dataframe, features, feature_means, feature_stds, verbose, inference_dim=(512, 512)):
        self.dataframe = dataframe.copy()
        
        self.image_folder = image_folder
        self.label_folder = label_folder

        # Sort by extracting numeric part of 'Filename'
        self.dataframe['sort_key'] = self.dataframe['Filename'].apply(
            lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
        self.dataframe = self.dataframe.sort_values('sort_key').reset_index(drop=True)
        self.dataframe.drop('sort_key', axis=1, inplace=True)  # Clean up the temporary column

        self.features = features
        
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        
        self.verbose = verbose
        
        self.n_actions = 4
        self.current_idx = 0

        self.PERFORMANCE_FACTOR = performance_factor
        
        self.network_performance_columns = ['0%', '25%', '50%', '75%']

        self.reset()

    def normalize_features(self, features):
        """Normalize the feature values."""
        return [(features[i] - self.feature_means[i]) / self.feature_stds[i] for i in range(len(features))]    
    
    def reset(self):
        """Point index at the beginning of the dataset (train or test)"""
        if self.current_idx >= len(self.dataframe):
            self.current_idx = 0

        print(self.dataframe)
        print(self.current_idx)
        current_features = self.dataframe.iloc[self.current_idx][self.features] 
        return self.normalize_features(current_features)

    def sample_action(self):
        """Randomly select an action."""
        return random.randint(0, self.n_actions - 1)

    def get_next_state_reward(self, action):
        """Determine the next state and reward after taking an action."""
        # Check if the selected action (network) was the best choice
        filename = self.dataframe.iloc[self.current_idx]['Filename']

        performances = {
            '0%':  self.dataframe.iloc[self.current_idx]['0%'],
            '25%': self.dataframe.iloc[self.current_idx]['25%'], 
            '50%': self.dataframe.iloc[self.current_idx]['50%'], 
            '75%': self.dataframe.iloc[self.current_idx]['75%']
        }

        # Sort the networks based on performance in descending order
        sorted_performances = sorted(performances.items(), key=lambda item: item[1], reverse=True)

        selected_network = self.network_performance_columns[action]
        # Extract just the network names from the sorted performances
        network_names_sorted = [network[0] for network in sorted_performances]
        
        if self.verbose:
            print(f"Filename: {filename}")
            print(f"selected_network: {selected_network}")

        # Find the index of the selected network in the sorted list
        selected_network_index = network_names_sorted.index(selected_network)
        if self.verbose:
            print(f"Index of selected network in sorted performances: {selected_network_index}")

        # The selected network's performance field name is assumed to be provided by 'action'
        selected_network_performance = self.dataframe.iloc[self.current_idx][selected_network]

        # Reward is the performance of the selected network
        network_to_normalized_weight = {
            "0%":  0.00,
            "25%": 0.25,
            "50%": 0.50,
            "75%": 0.75
        }

        #  reward = (selected_network_performance * self.PERFORMANCE_FACTOR) - (network_to_normalized_weight[selected_network] * self.WEIGHT_FACTOR)
        reward = (selected_network_performance * self.PERFORMANCE_FACTOR) + (1.0 - self.PERFORMANCE_FACTOR) * network_to_normalized_weight[selected_network]

        # Move to the next image (this could be random or sequential)
        self.current_idx = (self.current_idx + 1) % len(self.dataframe)

        current_features = self.dataframe.iloc[self.current_idx][self.features]
        next_state = self.normalize_features(current_features.tolist())

        # Check if the episode is done (if we've gone through all images)
        done = self.current_idx == 0

        return next_state, reward, done

    def has_next(self):
        """Check if there are more images to process."""
        return self.current_idx < len(self.dataframe) - 1

    def analyze_sequential_similarity(self, offsets=[1,2,3,4,5,10,20,50,100]):
        differences = {offset: {'sum_diff': 0, 'count': 0} for offset in offsets}

        for index in range(len(self.dataframe)):
            for offset in offsets:
                if index + offset < len(self.dataframe):
                    # Get current and offset feature vectors
                    current_features = self.dataframe.iloc[index][self.features]
                    offset_features = self.dataframe.iloc[index + offset][self.features]
                    
                    # Normalize both feature sets
                    normalized_current = self.normalize_features(current_features)
                    normalized_offset = self.normalize_features(offset_features)
                    
                    # Calculate Euclidean distance between the normalized features
                    diff = sum((nc - no) ** 2 for nc, no in zip(normalized_current, normalized_offset)) ** 0.5
                    
                    # Accumulate the differences and increment the count
                    differences[offset]['sum_diff'] += diff
                    differences[offset]['count'] += 1
        
        # Calculate average differences for each offset
        average_differences = {offset: differences[offset]['sum_diff'] / differences[offset]['count'] 
                               for offset in offsets if differences[offset]['count'] > 0}
        
        return average_differences