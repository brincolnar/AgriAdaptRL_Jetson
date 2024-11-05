import pandas as pd
import numpy as np

# Load your CSV data into a Pandas DataFrame
df = pd.read_csv('./features.csv')

# Drop non-numeric columns
df = df.drop(columns=['Filename'])

# Convert necessary columns to numeric type if they are not already
numeric_cols = ['Mean Brightness', 'Std Brightness', 'Max Brightness', 'Min Brightness',
                'Hue Hist Feature 1', 'Hue Hist Feature 2', 'Hue Std', 'Contrast',
                'Mean Saturation', 'Std Saturation', 'Max Saturation', 'Min Saturation',
                'SIFT Features', 'Texture Contrast', 'Texture Dissimilarity',
                'Texture Homogeneity', 'Texture Energy', 'Texture Correlation',
                'Texture ASM', 'Excess Green Index', 'Excess Red Index', 'CIVE',
                'ExG-ExR Ratio', 'CIVE Ratio', 'IoU Weeds', 'IoU Back']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Calculate correlation with respect to 'IoU Weeds' and 'IoU Back'
correlation_weeds = df.corr()['IoU Weeds']
correlation_back = df.corr()['IoU Back']

# Sort the features based on their absolute correlation values
sorted_correlation_weeds = correlation_weeds.abs().sort_values(ascending=False)
sorted_correlation_back = correlation_back.abs().sort_values(ascending=False)

print("Feature ranking with respect to IoU Weeds:")
print(sorted_correlation_weeds)
print("\nFeature ranking with respect to IoU Back:")
print(sorted_correlation_back)
