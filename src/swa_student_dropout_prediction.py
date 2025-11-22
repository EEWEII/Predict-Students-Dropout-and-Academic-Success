# Auto-install required packages
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of essential packages (pip names)
required_packages = [
    "pandas", "matplotlib", "seaborn", "scikit-learn", "xgboost"
]

for package in required_packages:
    install_if_missing(package)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Data Preprocessing
# Load dataset
df = pd.read_csv('data.csv', sep=';', encoding='utf-8-sig')

# Clean target variable, removes spaces, tabs, or newlines
df['Target'] = df['Target'].str.strip()

# Separate features and target BEFORE imputation
X = df.drop('Target', axis=1) #all the input data you will use to predict
y = df['Target'] #the output data you want to predict

# Handle missing values in FEATURES ONLY (Fills empty values with the median value)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Encode target variable (Converts words in the target into numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_encoded, test_size=0.2, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Machine Learning Modelling
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train) # Train the model
    y_pred = model.predict(X_test_scaled) # Make predictions on the test set
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# RESULT TABLE
# Create directory for visualizations
import os
os.makedirs('model_visualizations', exist_ok=True)

# Create separate subfolders for key metric visualizations
key_metric_dir = 'visualizations/key_metrics'
confusion_dir = 'visualizations/confusion_matrices'
distribution_dir = 'visualizations/probability_distributions'
training_dir = 'visualizations/training_comparison'

os.makedirs(key_metric_dir, exist_ok=True)
os.makedirs(confusion_dir, exist_ok=True)
os.makedirs(distribution_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

# Round results for display
table_data = results_df.round(4)

# Create figure and axis for the table
fig, ax = plt.subplots(figsize=(12, 2))  # Adjust size depending on row count
ax.axis('off')  # Hide axes

# Create the visual table
table = plt.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc='center',
    loc='center',
    colColours=['skyblue'] * len(table_data.columns)  # Header row color
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)  # Adjust scaling for better spacing

plt.savefig("model_visualizations/Performance_Metrics_Table.png", bbox_inches='tight', dpi=300)
plt.close()

# Model Performance Comparison (Individual Bar Charts)
for index, row in results_df.iterrows():
    plt.figure(figsize=(8, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    
    ax = sns.barplot(x=metrics, y=scores, hue=metrics, palette='viridis', legend=False)
    plt.title(f'{row["Model"]} Performance Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'model_visualizations/{row["Model"].replace(" ", "_")}_metrics.png', dpi=300)
    plt.close()

# Model Performance Comparison (Metric-wise Bar Charts)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
palette = sns.color_palette('Set2', len(models))

for metric in metrics:
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Model', y=metric, hue='Model', data=results_df, palette=palette, legend=False)

    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=9)
    
    plt.title(f'{metric} Comparison Across Models', fontsize=16)
    plt.ylabel(f'{metric} Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xlabel('Machine Learning Models', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/key_metrics/{metric}_comparison.png', dpi=300)
    plt.close()

# Individual Confusion Matrices
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_,
                annot_kws={"size": 12})
    plt.title(f'{name} Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/confusion_matrices/{name.replace(" ", "_")}_confusion_matrix.png', dpi=300)
    plt.close()

# Probability Distribution Histograms
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)
        plt.figure(figsize=(10, 8))
        
        # Create distribution plots for each class
        for class_idx, class_name in enumerate(le.classes_):
            sns.histplot(y_proba[:, class_idx], kde=True, bins=30, 
                         label=f'Class: {class_name}', alpha=0.6)
        
        plt.title(f'{name} - Predicted Probability Distributions', fontsize=16)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/probability_distributions/{name.replace(" ", "_")}_probability_dist.png', dpi=300)
        plt.close()

# Training Time Comparison
import time

training_times = []
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_times.append(time.time() - start_time)

# Create timing dataframe
time_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Training Time (s)': training_times
})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Model', y='Training Time (s)', hue='Model', data=time_df, palette='coolwarm', legend=False)
plt.title('Model Training Time Comparison', fontsize=16)
plt.ylabel('Seconds', fontsize=12)
plt.xlabel('Machine Learning Technique', fontsize=12)
plt.xticks(fontsize=10, rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/training_comparison/training_time_comparison.png', dpi=300)
plt.close()
print("All visualizations saved to 'model_visualizations' & 'visualizations' directory")