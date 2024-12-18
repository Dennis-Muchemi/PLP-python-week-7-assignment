# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# a. Load the dataset
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
iris_df = pd.DataFrame(data.data, columns=data.feature_names)
iris_df['species'] = data.target

# Map numerical species values to their names
species_mapping = {i: species for i, species in enumerate(data.target_names)}
iris_df['species'] = iris_df['species'].map(species_mapping)

# b. Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# c. Explore the structure of the dataset
print("\nDataset info:")
iris_df.info()

# Check for missing values
print("\nMissing values in the dataset:")
print(iris_df.isnull().sum())

# d. Clean the dataset (if necessary)
# The Iris dataset does not have missing values, but here is an example:
iris_df.fillna(method='ffill', inplace=True)

# Task 2: Basic Data Analysis
# a. Compute basic statistics
print("\nBasic statistics of numerical columns:")
print(iris_df.describe())

# b. Perform groupings and compute means
print("\nMean values grouped by species:")
print(iris_df.groupby('species').mean())

# Task 3: Data Visualization
sns.set(style="whitegrid")  # Use seaborn styles for plots

# a. Line chart (example using a cumulative sum of a column for demonstration)
plt.figure(figsize=(8, 6))
iris_df['cumulative_sepal_length'] = iris_df['sepal length (cm)'].cumsum()
plt.plot(iris_df.index, iris_df['cumulative_sepal_length'], label='Cumulative Sepal Length')
plt.title('Cumulative Sepal Length Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Sepal Length (cm)')
plt.legend()
plt.show()

# b. Bar chart (average petal length per species)
plt.figure(figsize=(8, 6))
iris_grouped = iris_df.groupby('species')['petal length (cm)'].mean()
iris_grouped.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# c. Histogram (distribution of sepal width)
plt.figure(figsize=(8, 6))
iris_df['sepal width (cm)'].plot(kind='hist', bins=15, color='purple', alpha=0.7)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# d. Scatter plot (sepal length vs. petal length)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='viridis')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Error handling example:
try:
    missing_dataset = pd.read_csv('non_existent_file.csv')
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")


