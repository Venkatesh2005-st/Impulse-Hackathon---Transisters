import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

#base directory containing class folders
base_dir = "C:\\college_stuff\\events\\impulse\\Impulse\\EEG_Data\\Train_data"

class_folders = [
    "Complex_Partial_Seizures",
    "Electrographic_Seizures",
    "Normal",
    "Video_detected_Seizures_with_no_visual_change_over_EEG"
]

#output directory
output_dir = "C:\\college_stuff\\events\\impulse\\task5.1\\output_plots"
os.makedirs(output_dir, exist_ok=True)


def plot_fourier(signal, sampling_rate, class_folder, file_name, class_output_dir):
    num_channels = signal.shape[0]
    for channel_idx in range(num_channels):
        # Extract the signal for the current channel
        channel_signal = signal[channel_idx]
        N = len(channel_signal)
        yf = fft(channel_signal)
        xf = fftfreq(N, 1 / sampling_rate)

        # Plot the Fourier Transform for the current channel
        plt.figure(figsize=(10, 5))
        plt.plot(xf[:N // 2], np.abs(yf[:N // 2]))
        plt.title(f"{class_folder}, Data point {file_name}, Channel {channel_idx + 1} - Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.savefig(os.path.join(class_output_dir, f"{file_name}_channel_{channel_idx + 1}.png"))
        plt.close()


sampling_rate = 500


#processing the files in each class folder
for class_folder in class_folders:
    class_path = os.path.join(base_dir, class_folder)
    if not os.path.isdir(class_path):
        print(f"Class folder not found: {class_path}")
        continue

    #list of .npy files
    npy_files = [f for f in os.listdir(class_path) if f.endswith(".npy")]

    #selecting the first file
    selected_file = npy_files[0]
    file_path = os.path.join(class_path, selected_file)
    signal = np.load(file_path)

    # creating an output subdirectory for the class
    class_output_dir = os.path.join(output_dir, class_folder)
    os.makedirs(class_output_dir, exist_ok=True)

    plot_fourier(signal, sampling_rate, class_folder, selected_file, class_output_dir)
    print(f"saved output to {class_output_dir}")