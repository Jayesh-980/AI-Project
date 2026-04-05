import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix, 
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE
print("--- Step 1: Data Setup ---")
df = pd.read_csv('fertility.csv') 

plt.figure(figsize=(6, 4))
sns.countplot(x=df.iloc[:, -1]) # Assuming target is the last column
plt.title("Original Class Distribution")
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
# Note: Target needs to be numeric for correlation, so we do a quick temporary conversion if it's not
temp_df = df.copy()
temp_df.iloc[:, -1] = temp_df.iloc[:, -1].map({'N': 0, 'O': 1}) # Adjust 'N'/'O' to your dataset's actual labels
sns.heatmap(temp_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================================
# STEP 2: Data Preprocessing
# ==========================================
print("\n--- Step 2: Data Preprocessing ---")
# Check missing values
print("Missing values per column:\n", df.isnull().sum())

# Encode target: Normal -> 0, Altered -> 1
# Update 'N' and 'O' to match the exact string values in your dataset's target column
target_col = df.columns[-1]
df[target_col] = df[target_col].map({'N': 0, 'O': 1}) 

# Encode Season using One-Hot Encoding (drop_first=True to avoid dummy trap)
# Assuming the column is named 'Season'
if 'Season' in df.columns:
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-test split: 80/20, random_state=42, stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# STEP 3: Handling Class Imbalance
# ==========================================
print("\n--- Step 3: Handling Class Imbalance ---")
# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Plot class distribution before vs after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title("Training Class Distribution BEFORE SMOTE")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Normal (0)', 'Altered (1)'])

sns.countplot(x=y_train_smote, ax=axes[1])
axes[1].set_title("Training Class Distribution AFTER SMOTE")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Normal (0)', 'Altered (1)'])
plt.show()

# ==========================================
# STEP 4: Model Development
# ==========================================
print("\n--- Step 4: Model Development ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5]
}

# GridSearch with Random Forest
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid.fit(X_train_smote, y_train_smote)

best_model = grid.best_estimator_
print(f"Best Parameters from GridSearchCV: {grid.best_params_}")

# Extra unique graph: Horizontal bar chart of feature importances
importances = best_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.title("Feature Importances (Best Random Forest Model)")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# ==========================================
# STEP 5: Model Evaluation
# ==========================================
print("\n--- Step 5: Model Evaluation ---")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

# Print metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
acc_tuned = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc_tuned:.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal (0)', 'Altered (1)'], yticklabels=['Normal (0)', 'Altered (1)'])
plt.title("Confusion Matrix (Tuned Random Forest)")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC curve with AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# ==========================================
# STEP 6: Model Comparison
# ==========================================
print("\n--- Step 6: Model Comparison ---")
# Train default Random Forest (n_estimators=100) on SMOTE data
default_rf = RandomForestClassifier(n_estimators=100, random_state=42)
default_rf.fit(X_train_smote, y_train_smote)
y_pred_default = default_rf.predict(X_test)
acc_default = accuracy_score(y_test, y_pred_default)

print(f"Default Random Forest Accuracy: {acc_default:.4f}")
print(f"Tuned Random Forest Accuracy: {acc_tuned:.4f}")

# Bar chart showing accuracy of both models
models = ['Default Random Forest', 'GridSearch Tuned Random Forest']
accuracies = [acc_default, acc_tuned]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['gray', 'green'])
plt.ylim(0, 1)
plt.ylabel('Accuracy Score')
plt.title('Model Comparison: Default vs. Tuned Random Forest')

# Add accuracy values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

plt.show()