import os
import json
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Directory containing the JSON files
directory = './metaseg'

# Lists to store metric values
iou_values = []
E_values = []
E_in_values = []
E_bd_values = []
D_in_values = []
D_bd_values = []
D_values = []

# Load data from each JSON file
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath) as file:
            data = json.load(file)
            iou_values.extend(data["iou"])
            E_values.extend(data["E"])
            E_in_values.extend(data["E_in"])
            E_bd_values.extend(data["E_bd"])
            D_in_values.extend(data["D_in"])
            D_bd_values.extend(data["D_bd"])
            D_values.extend(data["D"])

# Convert lists to numpy arrays
iou_values = np.array(iou_values)
E_values = np.array(E_values)
E_in_values = np.array(E_in_values)
E_bd_values = np.array(E_bd_values)
D_in_values = np.array(D_in_values)
D_bd_values = np.array(D_bd_values)
D_values = np.array(D_values)

# Define a function to calculate and print correlation and p-value
def correlation_analysis(x, y, metric_name):
    correlation, p_value = stats.pearsonr(x, y)
    print(f"Correlation between IoU and {metric_name}: {correlation:.4f}")
    print(f"P-value for correlation between IoU and {metric_name}: {p_value:.4g}\n")

# Perform correlation analysis
correlation_analysis(iou_values, E_values, "E")
correlation_analysis(iou_values, E_in_values, "E_in")
correlation_analysis(iou_values, E_bd_values, "E_bd")
correlation_analysis(iou_values, D_in_values, "D_in")
correlation_analysis(iou_values, D_bd_values, "D_bd")
correlation_analysis(iou_values, D_values, "D")


# Convert lists to numpy arrays
E_values = E_values.reshape(-1, 1)
E_in_values = E_in_values.reshape(-1, 1)
E_bd_values = E_bd_values.reshape(-1, 1)
D_in_values = D_in_values.reshape(-1, 1)
D_bd_values = D_bd_values.reshape(-1, 1)
D_values = D_values.reshape(-1, 1)

# Define a function to fit linear regression and print R² and prediction std deviation
def linear_regression_analysis(X, y, feature_name):
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate R² score
    r2 = r2_score(y, predictions)
    
    # Calculate standard deviation of residuals (prediction error)
    residuals = y - predictions
    std_dev = np.std(residuals)
    
    print(f"Linear Regression for {feature_name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"Standard Deviation of Prediction Errors: {std_dev:.4f}\n")

# Perform linear regression analysis for each feature
linear_regression_analysis(E_values, iou_values, "E")
linear_regression_analysis(E_in_values, iou_values, "E_in")
linear_regression_analysis(E_bd_values, iou_values, "E_bd")
linear_regression_analysis(D_in_values, iou_values, "D_in")
linear_regression_analysis(D_bd_values, iou_values, "D_bd")
linear_regression_analysis(D_values, iou_values, "D")
