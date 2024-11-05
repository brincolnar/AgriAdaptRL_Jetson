import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from calculate_oracle import OracleComputation
import matplotlib.pyplot as plt
import seaborn as sns

features = pd.read_csv("./features.csv")
chosen_features = ['Max Saturation', 'Hue Hist Feature 2', 'Mean Brightness']

def analyze_variance():
    numeric = features.select_dtypes(include=[np.number])

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(numeric)

    scaled_df = pd.DataFrame(scaled, columns=numeric.columns)

    variances_scaled = scaled_df.var()

    top_3_features_scaled = variances_scaled.sort_values(ascending=False).head(3).index.tolist()

    print("top_3_features_scaled")
    print(top_3_features_scaled)

    print("variances: ")
    print(variances_scaled.sort_values(ascending=False))

def quantile_discretization(num_of_bins):
    for feature in chosen_features:
        feature_ = features[feature]

        sorted_features_vals = feature_.sort_values()

        quantiles = np.linspace(0, 1, num_of_bins)[1:-1]  # Exclude 0 and 1

        # print("quantiles: ")
        # print(quantiles)

        split_points = sorted_features_vals.quantile(quantiles)

        subintervals = pd.cut(sorted_features_vals, bins=[-np.inf] + split_points.tolist() + [np.inf], labels=False)

        features[feature + ' Interval'] = subintervals

        # print("Split Points:")
        # print(split_points)

        print("\nUpdated DataFrame with Subintervals:")
        print(features[[feature, feature + ' Interval']].head(10))
    features.to_csv("features.csv", index=False)

def write_image_type_preformances():

    si = OracleComputation(
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            256,
            256,
        ),
        # slim or squeeze
        model_architecture="squeeze",
        dataset="geok",
        preformance_analysis=True,
        # Do you want to generate a mask/image overlay
        save_image=False,
        # Was segmentation model trained using transfer learning
        is_trans=True,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,
    )
    
    network1_preformances = []
    network2_preformances = []
    network3_preformances = []
    network4_preformances = []

    for index, row in features.iterrows():
        path = row['Filename']
        print(path)
        preformances = si.infer(path)
        print("preformances")
        print(preformances)
        
        # add preformances 
        network1_preformances.append(preformances[0.25])
        network2_preformances.append(preformances[0.5])
        network3_preformances.append(preformances[0.75])
        network4_preformances.append(preformances[1.0])

    if si.model_architecture == "squeeze":
        features['ActualSqueezeNetwork1Preformance'] = pd.Series(network1_preformances)
        features['ActualSqueezeNetwork2Preformance'] = pd.Series(network2_preformances)
        features['ActualSqueezeNetwork3Preformance'] = pd.Series(network3_preformances)
        features['ActualSqueezeNetwork4Preformance'] = pd.Series(network4_preformances)
    elif si.model_architecture == "slim":
        features['ActualSlimNetwork1Preformance'] = pd.Series(network1_preformances)
        features['ActualSlimNetwork2Preformance'] = pd.Series(network2_preformances)
        features['ActualSlimNetwork3Preformance'] = pd.Series(network3_preformances)
        features['ActualSlimNetwork4Preformance'] = pd.Series(network4_preformances)
    print(features.head())

    save = True

    if save:
        features.to_csv('features.csv', index=False)


def analyze_image_type_preformances():
    # group rows by Max Saturation Interval,Hue Hist Feature 2 Interval
    grouped = features.groupby(['Max Saturation Interval', 'Hue Hist Feature 2 Interval', 'Mean Brightness Interval'])

    for (max_sat, hue_hist, mean_bright), group in grouped:
        averages = group[['ActualSlimNetwork1Preformance', 'ActualSlimNetwork2Preformance', 'ActualSlimNetwork3Preformance', 'ActualSlimNetwork4Preformance', 'ActualSqueezeNetwork1Preformance', 'ActualSqueezeNetwork2Preformance', 'ActualSqueezeNetwork3Preformance', 'ActualSqueezeNetwork4Preformance']].mean()
        
        group_size = group.shape[0]

        print(f"Group - Max Saturation: {max_sat}, Hue Hist Feature 2: {hue_hist}, Mean Brightness: {mean_bright}")
        print(f"Number of rows in this group: {group_size}")
        print("Averages:")
        print(averages)
        print("\n")  # Newline for better readability

def model_usage_versus_feature(feature_name, features, save_path):
    # Calculate quartiles for the feature
    quartiles = np.quantile(features[feature_name], [0.25, 0.5, 0.75])
    
    # Determine the quartile for each row
    conditions = [
        (features[feature_name] <= quartiles[0]),
        (features[feature_name] > quartiles[0]) & (features[feature_name] <= quartiles[1]),
        (features[feature_name] > quartiles[1]) & (features[feature_name] <= quartiles[2]),
        (features[feature_name] > quartiles[2])
    ]
    choices = ['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile']
    features['Quartile'] = np.select(conditions, choices)

    performance_columns = [
        'ActualSlimNetwork1Preformance', 'ActualSlimNetwork2Preformance', 
        'ActualSlimNetwork3Preformance', 'ActualSlimNetwork4Preformance',
        'ActualSqueezeNetwork1Preformance', 'ActualSqueezeNetwork2Preformance',
        'ActualSqueezeNetwork3Preformance', 'ActualSqueezeNetwork4Preformance'
    ]

    # Preparing data for the plot
    performance_means = {col: [] for col in performance_columns}
    for quartile in choices:
        quartile_data = features[features['Quartile'] == quartile]
        for col in performance_columns:
            if quartile_data[col].size > 0:
                average_performance = quartile_data[col].mean()
                performance_means[col].append(average_performance)
            else:
                performance_means[col].append(np.nan)

    # Plotting with a larger figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate the width of each bar so that they all fit within the quartile grouping
    n_groups = len(choices)
    n_bars = len(performance_columns)
    total_width = 0.8  # Total space allocated for all bars in one group
    bar_width = total_width / n_bars  # Individual bar width
    x = np.arange(len(choices))  # The label locations

    # Calculate the x offset for each bar
    bar_offsets = (np.arange(n_bars) - n_bars / 2) * bar_width + bar_width / 2

    for i, col in enumerate(performance_columns):
        means = performance_means[col]
        if not all(np.isnan(means)):
            # Adjust the bar's x position based on its offset
            ax.bar(x + bar_offsets[i], means, width=bar_width, label=col)

    # Define the x-axis labels and their positions
    ax.set_xlabel('Quartiles')
    ax.set_ylabel('Average Performance')
    ax.set_title(f'Average Performance per Quartile for {feature_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(choices)
    # Place the legend outside of the figure/plot
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')

    # Adjust layout to make room for the legend and prevent clipping of tick-labels
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the right boundary of the layout

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')  # Ensure the entire plot is saved, including the legend

    # Optionally, close the plot to free up memory
    plt.close(fig)


def feature_preformance_correlation(features):
    # Drop non-numeric columns
    numeric_features = features.select_dtypes(include=[np.number])

    # Calculate correlation matrix for numeric features only
    correlation_matrix = numeric_features.corr()

    # Select only the columns for performance
    performance_columns = ['ActualSlimNetwork1Preformance', 'ActualSlimNetwork2Preformance', 
                        'ActualSlimNetwork3Preformance', 'ActualSlimNetwork4Preformance', 
                        'ActualSqueezeNetwork1Preformance', 'ActualSqueezeNetwork2Preformance', 
                        'ActualSqueezeNetwork3Preformance', 'ActualSqueezeNetwork4Preformance']

    # Filter the correlation matrix to show only correlations with performance columns
    performance_correlation = correlation_matrix[performance_columns].drop(performance_columns)

    # Plotting the correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_correlation, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation with Performance')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    plt.savefig('feature_correlation_with_performance.png')

    # Close the plot
    plt.close()

# quantile_discretization(4)
# analyze_image_type_preformances()
for feature in chosen_features:
    model_usage_versus_feature(feature, features, f'./graphs/{feature}_plot.png')
    model_usage_versus_feature(feature, features, f'./graphs/{feature}_plot.png')
    model_usage_versus_feature(feature, features, f'./graphs/{feature}_plot.png')
# feature_preformance_correlation(features)