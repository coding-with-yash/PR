import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Read the dataset
df = pd.read_csv("/content/iris.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.drop(columns=['variety'])

# Fill missing values with mean or other strategies
df_filled = df.fillna(df.mean())

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df_no_duplicates = df.drop_duplicates()

# Check data types
print(df_cleaned.dtypes)

# Data Analysis
print(df_cleaned.describe())

# Calculate correlation matrix
correlation_matrix = df_cleaned.corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, cmap='viridis')
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()

# Perform t-test or other statistical tests
# Note: Replace 'group1' and 'group2' with your actual column names
t_stat, p_value = stats.ttest_ind(df_cleaned['sepal.length'], df_cleaned['sepal.width'])
print(f'T-statistic: {t_stat}, p-value: {p_value}')

# Plot histogram
plt.hist(df_cleaned['sepal.length'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of sepal.length')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.scatter(df['petal.length'], df['petal.width'], color='red', alpha=0.5)
plt.title('Scatter Plot between petal.length and petal.width')
plt.xlabel('petal.length')
plt.ylabel('petal.width')
plt.show()
