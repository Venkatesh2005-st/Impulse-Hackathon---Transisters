import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

# Base directory containing class folders
base_dir = "C:\\college_stuff\\events\\impulse\\task6.1\\outputs"

class_folders = [
    "Complex_Partial_Seizures",
    "Electrographic_Seizures",
    "Normal",
    "Video_detected_Seizures_with_no_visual_change_over_EEG"
]

outputs = ["fourier_outputs", "metrics_output"]

# Initialize empty DataFrame
train_set_df = pd.DataFrame()

for class_folder in class_folders:
    class_path = os.path.join(base_dir, class_folder)
    if not os.path.isdir(class_path):
        print(f"Class folder not found: {class_path}")
        continue

    fourier_df = None  # To store Fourier transform data
    metrics_df = None  # To store metric data

    for output in outputs:
        output_dir = os.path.join(class_path, output)
        if not os.path.exists(output_dir):
            print(f"Output directory not found: {output_dir}")
            continue

        csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
        if not csv_files:
            print(f"No CSV files found in {output_dir}")
            continue

        # Load the first CSV file (or all if needed)
        file_path = os.path.join(output_dir, csv_files[0])
        df = pd.read_csv(file_path)

        if output == "fourier_outputs":
            df["class"] = class_folder  # Add class label
            fourier_df = df
        else:  # "metrics_output"
            metrics_df = df[["Zero_Crossing_Rate"]]  # Only keep needed column

    # Merge Fourier and Metrics data
    if fourier_df is not None and metrics_df is not None:
        fourier_df = fourier_df.reset_index(drop=True)
        metrics_df = metrics_df.reset_index(drop=True)
        merged_df = pd.concat([fourier_df, metrics_df], axis=1)
        train_set_df = pd.concat([train_set_df, merged_df], ignore_index=True)

# Ensure class column exists before splitting
if "class" not in train_set_df.columns:
    raise ValueError("Class column missing in the dataset!")



# Prepare data for training
X = train_set_df.drop(columns=["class"])
X = X.fillna(0)  # Handle NaNs
y = train_set_df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

# Train SVM model
model = SVC(probability= True)
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print("Accuracy Score:", score)

probabilities = model.predict_proba(X_test)[:, 1]
#roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr")
#print("roc: ",roc_auc)
balanced_acc = balanced_accuracy_score(y_test, predictions)
print("balanced: ",balanced_acc)
