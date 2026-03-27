
#Data Preprocessing - Step 3.2


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



# Load the dataset
df = pd.read_csv('fertility-1.csv')
print("\nDataset loaded successfully!")

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# No missing values in this dataset
print("\nNo missing values found!")

# Encode the target variable
print("\nEncoding target variable...")
encoder = LabelEncoder()
df['output_encoded'] = encoder.fit_transform(df['output'])

print("Original values:", df['output'].unique())
print("Encoded values:", df['output_encoded'].unique())
print("Mapping: N -> 0 (Normal), O -> 1 (Altered)")

# Separate features and target
print("\nSeparating features and target...")
X = df.drop(['output', 'output_encoded'], axis=1)
y = df['output_encoded']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {list(X.columns)}")

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Feature statistics after standardization:")
print(X_scaled.describe().loc[['mean', 'std']])

# Split into train and test sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Check class distribution in train and test
print("\nClass distribution in training set:")
train_counts = y_train.value_counts()
for label, count in train_counts.items():
    if label == 0:
        class_name = "Normal"
    else:
        class_name = "Altered"
    percentage = (count / len(y_train)) * 100
    print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

print("\nClass distribution in testing set:")
test_counts = y_test.value_counts()
for label, count in test_counts.items():
    if label == 0:
        class_name = "Normal"
    else:
        class_name = "Altered"
    percentage = (count / len(y_test)) * 100
    print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

# Save preprocessed data for later use
print("\nSaving preprocessed data...")
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")


print("\nPreprocessing Summary:")
print(f"Total samples: {len(df)}")
print(f"Features after preprocessing: {X_scaled.shape[1]}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("Features standardized (mean=0, std=1)")
print("Target encoded (0=Normal, 1=Altered)")
