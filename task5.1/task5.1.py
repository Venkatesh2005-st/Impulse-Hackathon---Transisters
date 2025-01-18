import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt

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


def plot_fourier(signal, sampling_rate, class_folder, file_name, fourier_output_dir):
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
        plt.savefig(os.path.join(fourier_output_dir, f"{file_name}_channel_{channel_idx + 1}.png"))
        plt.close()


# Perform 4-level Wavelet Decomposition for each channel
def wavelet_decomposition(signal, wavelet='db4', level=4):
    coeffs_multichannel = [pywt.wavedec(signal[channel], wavelet, level=level) for channel in range(signal.shape[0])]
    return coeffs_multichannel  # Returns a list of coefficients for each channel


# Plot coefficients for all channels
def plot_wavelet_coeffs(coeffs_multichannel, class_folder, selected_file, wavelet_output_dir):
    num_channels = len(coeffs_multichannel)
    for channel_idx, coeffs in enumerate(coeffs_multichannel):
        plt.figure(figsize=(12, 8))
        for i, coeff in enumerate(coeffs):
            plt.subplot(len(coeffs), 1, i + 1)
            plt.plot(coeff)
            plt.title(f"{class_folder}, Data point {selected_file}, Channel {channel_idx + 1} - Level {i} {'Approximation' if i == 0 else 'Detail'} Coefficients")
        plt.tight_layout()
        plt.savefig(os.path.join(wavelet_output_dir, f"{selected_file}_channel_{channel_idx + 1}.png"))
        plt.close()


# Find the coefficient most similar to the original signal (e.g., using correlation)
def find_most_similar_coeff(signal, coeffs):
    correlations = [np.corrcoef(signal, coeff[:len(signal)])[0, 1] for coeff in coeffs]
    most_similar_level = np.argmax(correlations)
    return most_similar_level, correlations



sampling_rate = 500
wavelet = 'db4'

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
    file_name = selected_file.split(".")[0]
    signal = np.load(file_path)

    # creating an output subdirectory for the class
    class_output_dir = os.path.join(output_dir, class_folder)
    os.makedirs(class_output_dir, exist_ok=True)
    fourier_output_dir = os.path.join(class_output_dir, "fourier_transforms")
    os.makedirs(fourier_output_dir, exist_ok=True)
    wavelet_output_dir = os.path.join(class_output_dir, "wavelet_decompositions")
    os.makedirs(wavelet_output_dir, exist_ok=True)

    coeffs = wavelet_decomposition(signal, wavelet, 4)

    plot_fourier(signal, sampling_rate, class_folder, file_name, fourier_output_dir)
    plot_wavelet_coeffs(coeffs, class_folder, file_name, wavelet_output_dir)
    most_similar_level, correlations = find_most_similar_coeff(signal, coeffs)
    print(f"Most similar coefficient level: {most_similar_level}")
    print(f"Correlations: {correlations}")
    print(f"saved output to {class_output_dir}")