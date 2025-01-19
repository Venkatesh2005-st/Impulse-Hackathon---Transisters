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

class_mapping = {
    "Complex_Partial_Seizures" : 0,
    "Electrographic_Seizures" : 1,
    "Normal" : 2,
    "Video_detected_Seizures_with_no_visual_change_over_EEG" : 3
}

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

    output_dir = os.path.join(class_path, "fourier_outputs")
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        continue

    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
    if not csv_files:
        print(f"No CSV files found in {output_dir}")
        continue

    # Load the first CSV file (or all if needed)
    for csv_file in csv_files:
        file_path = os.path.join(output_dir, csv_file)
        df = pd.read_csv(file_path)

        train_set_df = pd.concat([train_set_df, df])
    print('.')

# Ensure class column exists before splitting
if "class" not in train_set_df.columns:
    raise ValueError("Class column missing in the dataset!")



train_set_df.to_csv('C:\\college_stuff\\events\\impulse\\task6.1\\train_set_df.csv', index=False)

# Prepare data for training
X = train_set_df.drop(columns=["class"])
X = X.fillna(0)  # Handle NaNs
y = train_set_df["class"]

X.to_csv('C:\\college_stuff\\events\\impulse\\task6.1\\X.csv', index=False)
y.to_csv('C:\\college_stuff\\events\\impulse\\task6.1\\y.csv', index=False)