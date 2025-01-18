import os
import numpy as np
import pandas as pd

#base directory containing class folders
base_dir = "C:\\college_stuff\\events\\impulse\\Impulse\\EEG_Data\\test_data"

#output directory
output_dir = "C:\\college_stuff\\events\\impulse\\task6.1\\test_data_csv"
os.makedirs(output_dir, exist_ok=True)

#list of .npy files
npy_files = [f for f in os.listdir(base_dir) if f.endswith(".npy")]

# creating an output subdirectory for the class
os.makedirs(output_dir, exist_ok=True)

#selecting the first file
for selected_file in npy_files:
    file_path = os.path.join(base_dir, selected_file)
    file_name = selected_file.split(".")[0]
    signal = np.load(file_path)

    # Create a DataFrame from the features
    signal_df = pd.DataFrame(signal)

    # Save to CSV
    #output_file = 'fourier_features.csv'
    file_output = os.path.join(output_dir,f"{file_name}.csv")
    signal_df.to_csv(file_output, index=False)
    #print(f"Features saved to {output_file}")

print(f"saved output to {output_dir}")