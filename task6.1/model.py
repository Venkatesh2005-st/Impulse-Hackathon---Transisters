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

train_set_df = pd.read_csv('C:\\college_stuff\\events\\impulse\\task6.1\\train_set_df.csv')

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
roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr")
print("roc: ",roc_auc)
balanced_acc = balanced_accuracy_score(y_test, predictions)
print("balanced: ",balanced_acc)
