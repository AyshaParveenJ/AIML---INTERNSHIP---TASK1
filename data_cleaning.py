# Import the necessary tools (libraries)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load the data and explore it ---
# Load your dataset from the CSV file
df = pd.read_csv('Titanic-Dataset.csv')

# Print basic info to see the columns, data types, and missing values
print("Initial Data Information:")
df.info()

# --- Step 2: Handle missing values and categorical features ---
# Find the median Age to fill in missing values
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Find the most frequent value for 'Embarked' and fill missing values
most_frequent_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_frequent_embarked, inplace=True)

# Remove the 'Cabin' column because it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Convert categorical columns ('Sex' and 'Embarked') into numerical form
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Print the updated info to see the changes
print("\nData Information after handling missing values and converting text:")
df.info()

# --- Step 3: Scale numerical features ---
# Select the numerical columns to be scaled
numerical_features = ['Age', 'Fare']
scaler = StandardScaler()

# Apply the scaling
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nSample of scaled numerical data (Age and Fare):")
print(df[numerical_features].head())

# --- Step 4: Visualize outliers ---
# Create a boxplot to see any outliers in the 'Fare' column
print("\nDisplaying a boxplot for 'Fare' to visualize outliers...")
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Box Plot of Fare')
plt.show()

print("\nData cleaning and preprocessing is complete. The dataset is now ready for machine learning.")