import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("STEP 3.1: DATA SETUP & EXPLORATION")
print("="*60)

# Load the dataset
df = pd.read_csv('fertility-1.csv')
print("\nDataset loaded successfully!")

# Check basic information
print("\nDataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nColumn Names:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check class distribution
print("\nClass Distribution:")
class_counts = df['output'].value_counts()
print(class_counts)

total = len(df)
for label, count in class_counts.items():
    if label == 'N':
        status = "Normal"
    else:
        status = "Altered"
    percentage = (count / total) * 100
    print(f"{label} ({status}): {count} samples ({percentage:.2f}%)")

# Check unique values in each feature
print("\nUnique values in each feature:")
for col in df.columns:
    if col != 'output':
        unique_vals = df[col].unique()
        print(f"{col}: {sorted(unique_vals)}")

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Create visualizations
print("\nCreating visualizations...")

# Plot 1: Class distribution
plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Fertility Status')
plt.xlabel('Fertility Status')
plt.ylabel('Number of Samples')
plt.xticks(ticks=[0, 1], labels=['Normal (N)', 'Altered (O)'], rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
print("Saved: class_distribution.png")

# Plot 2: Age distribution with actual years
df['actual_age'] = 18 + (df['age'] * 32)

plt.figure(figsize=(8, 5))
plt.hist(df['actual_age'], bins=15, color='blue', alpha=0.7, edgecolor='black')
plt.title('Age Distribution of Participants', fontsize=14)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Number of Participants', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('age_distribution.png')
print("Saved: age_distribution.png (shows actual ages 34-50 years)")

# Plot 3: Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.drop('output', axis=1)
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved: correlation_heatmap.png")


print(f"\nTotal samples: {len(df)}")
print(f"Total features: {len(df.columns)-1}")
print(f"Normal samples: {class_counts.get('N', 0)}")
print(f"Altered samples: {class_counts.get('O', 0)}")
print(f"Age range: {df['actual_age'].min():.1f} to {df['actual_age'].max():.1f} years")
