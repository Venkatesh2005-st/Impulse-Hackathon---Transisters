import os
import numpy as np
import pandas as pd

#base directory containing class folders
base_dir = "C:\\college_stuff\\events\\impulse\\Impulse\\EEG_Data\\validation_data"

class_folders = [
    "Complex_Partial_Seizures",
    "Electrographic_Seizures",
    "Normal",
    "Video_detected_Seizures_with_no_visual_change_over_EEG"
]

#output directory
output_dir = "C:\\college_stuff\\events\\impulse\\task6.1\\validation_data_csv"
os.makedirs(output_dir, exist_ok=True)



#processing the files in each class folder
for class_folder in class_folders:
    class_path = os.path.join(base_dir, class_folder)
    if not os.path.isdir(class_path):
        print(f"Class folder not found: {class_path}")
        continue

    #list of .npy files
    npy_files = [f for f in os.listdir(class_path) if f.endswith(".npy")]

    # creating an output subdirectory for the class
    class_output_dir = os.path.join(output_dir, class_folder)
    os.makedirs(class_output_dir, exist_ok=True)

    #selecting the first file
    for selected_file in npy_files:
        file_path = os.path.join(class_path, selected_file)
        file_name = selected_file.split(".")[0]
        signal = np.load(file_path)

        # Create a DataFrame from the features
        signal_df = pd.DataFrame(signal)

        # Save to CSV
        #output_file = 'fourier_features.csv'
        file_output = os.path.join(class_output_dir,f"{file_name}.csv")
        signal_df.to_csv(file_output, index=False)
        #print(f"Features saved to {output_file}")

    print(f"saved output to {class_output_dir}")