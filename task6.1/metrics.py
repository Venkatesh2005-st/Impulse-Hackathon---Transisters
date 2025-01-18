import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#base directory containing class folders
base_dir = "C:\\college_stuff\\events\\impulse\\Impulse\\EEG_Data\\Train_data"

class_folders = [
    "Complex_Partial_Seizures",
    "Electrographic_Seizures",
    "Normal",
    "Video_detected_Seizures_with_no_visual_change_over_EEG"
]

#output directory
output_dir = "C:\\college_stuff\\events\\impulse\\task4.1\\output_plots"
os.makedirs(output_dir, exist_ok=True)

#function to compute time domain metrics for EEG data
def compute_metrics(eeg_data):
    metrics = []
    for channel_idx, channel_data in enumerate(eeg_data):
        mean_val = np.mean(channel_data)
        zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
        range_val = np.ptp(channel_data)
        energy = np.sum(channel_data ** 2)
        rms = np.sqrt(np.mean(channel_data ** 2))
        variance = np.var(channel_data)

        metrics.append({
            "Channel": channel_idx + 1,
            "Mean": mean_val,
            "Zero_Crossing_Rate": zero_crossings,
            "Range": range_val,
            "Energy": energy,
            "RMS": rms,
            "Variance": variance
        })
    return metrics



#function to save metrics
def save_metrics(file_path, output_dir, class_folder):
    eeg_data = np.load(file_path)

    file_name = os.path.basename(file_path).split(".")[0]

    #computing metrics
    metrics = compute_metrics(eeg_data)

    metrics_df = pd.DataFrame(metrics)

    # Save metrics to a CSV file
    metrics_file_path = os.path.join(output_dir, f"{file_name}_metrics.csv")
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"Metrics saved to: {metrics_file_path}")


#processing the files in each class folder
for class_folder in class_folders:
    class_path = os.path.join(base_dir, class_folder)
    if not os.path.isdir(class_path):
        print(f"Class folder not found: {class_path}")
        continue

    #list of .npy files
    npy_files = [f for f in os.listdir(class_path) if f.endswith(".npy")]

    #selecting the first file
    for selected_file in npy_files:
        file_path = os.path.join(class_path, selected_file)

        #creating an output subdirectory for the class
        class_output_dir = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_dir, exist_ok=True)

        #plotting the signals and computing metrics for the selected file
        plot_signals(file_path, class_output_dir, class_folder)

print("Plots and metrics have been generated and saved in the output directory.")