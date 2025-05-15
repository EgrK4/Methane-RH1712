import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pykalman import KalmanFilter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft

# Load your dataset
# METH = pd.read_csv('methane_data.csv')
# print(f"Columns: {METH.columns.tolist()}")


# Keep only the first 6 columns and the n+1th column
def pick_sensor(n):
    METH = pd.read_csv('methane_data.csv')
    METH = METH.iloc[:, list(range(6)) + [n]]
    return METH

# Display basic information
# print(f"Dataset shape: {METH.shape}")  # (rows, columns)
# print("\nFirst 5 rows:")
# print(METH.head(5))
# print("\nData types:")
# print(METH.dtypes)

# Unfiltered RH1712 plot
def plot_unfiltered(DATAFRAME):
    plt.figure(figsize=(25, 6))
    plt.title('Нефильтрованный сигнал')
    plt.plot(DATAFRAME.iloc[:, 6], color='blue', linewidth=0.7)

# Savitzky-Golay filter RH1712 plot
def plot_savitzky_golay_filter(DATAFRAME):
    filtered_signal = savgol_filter(DATAFRAME.iloc[:, 6], window_length=101, polyorder=3)
    plt.figure(figsize=(25, 6))
    plt.plot(DATAFRAME.iloc[:, 6], label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал', color='blue', linewidth=0.7)
    plt.title('Фильтрация фильтром Савицкого-Голая')
    plt.legend()

from scipy.fft import fft, ifft

# Fourier Transform filter RH1712 plot
def plot_fourier_filter(DATAFRAME):
    signal = DATAFRAME.iloc[:, 6].dropna().values
    fft_signal = fft(signal)
    fft_signal[int(len(fft_signal) * 0.1):] = 0  # Zero out high frequencies
    filtered_signal = ifft(fft_signal).real
    plt.figure(figsize=(25, 6))
    plt.plot(signal, label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал (Fourier)', color='blue', linewidth=0.7)
    plt.title('Фильтрация с использованием преобразования Фурье')
    plt.legend()

# Moving average RH1712 plot
def plot_moving_average(DATAFRAME):
    filtered_signal = DATAFRAME.iloc[:, 6].rolling(window=10000, center=True).median()
    plt.figure(figsize=(25, 6))
    plt.plot(DATAFRAME.iloc[:, 6], label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал', color='blue', linewidth=0.7)
    plt.title('Фильтрация скользящим средним')
    plt.legend() 

# Exponential moving average RH1712 plot
def plot_exponential_moving_average(DATAFRAME):
    filtered_signal = DATAFRAME.iloc[:, 6].ewm(alpha=0.2, adjust=False).mean()
    plt.figure(figsize=(25, 6))
    plt.plot(DATAFRAME.iloc[:, 6], label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал', color='blue', linewidth=0.7)
    plt.title('Фильтрация экспоненциальным фильтром')
    plt.legend()

# Double exponential moving average RH1712 plot
def plot_double_exponential_moving_average(DATAFRAME):
    filtered_signal = DATAFRAME.iloc[:, 6].ewm(alpha=0.2, adjust=False).mean()
    filtered_signal = filtered_signal.ewm(alpha=0.03, adjust=False).mean()
    plt.figure(figsize=(25, 6))
    plt.plot(DATAFRAME.iloc[:, 6], label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал', color='blue', linewidth=0.7)
    plt.title('Фильтрация двойным экспоненциальным фильтром')
    plt.legend()

# Gaussian filter RH1712 plot
def plot_gaussian_filter(DATAFRAME):
    filtered_signal = gaussian_filter1d(DATAFRAME.iloc[:, 6], sigma=100)
    plt.figure(figsize=(25, 6))
    plt.plot(DATAFRAME.iloc[:, 6], label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал', color='blue', linewidth=0.7)
    plt.title('Фильтрация гауссовым фильтром')
    plt.legend()

# Kalman filter plot
def plot_kalman_filter(DATAFRAME):
    signal = DATAFRAME.iloc[:, 6].dropna().values  # Ensure no NaN values
    if len(signal) == 0:
        print("Error: Signal is empty or contains only NaN values.")
        return

    # Initialize the Kalman filter
    kf = KalmanFilter(initial_state_mean=signal[0], n_dim_obs=1)
    try:
        kf = kf.em(signal, n_iter=10)  # Estimate parameters
        filtered_signal, _ = kf.filter(signal)
    except Exception as e:
        print(f"Error applying Kalman filter: {e}")
        return

    # Plot the results
    plt.figure(figsize=(25, 6))
    plt.plot(signal, label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал (Kalman)', color='blue', linewidth=0.7)
    plt.title('Фильтрация фильтром Калмана')
    plt.legend()
    
    plt.figure(figsize=(25, 6))
    plt.plot(signal, label='нефильтрованный сигнал', color='red', linewidth=0.7)
    plt.plot(filtered_signal, label='отфильтрованный сигнал (Kalman)', color='blue', linewidth=0.7)
    plt.title('Фильтрация фильтром Калмана')
    plt.legend()
    
def finishing_touches(value):
    # Hide x-axis values
    plt.xticks([])

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel(value)
    plt.show()

# Main function
print("Введи индекс сенсора (от 6 до 33):")
sensor_index = int(input())
print("Что измеряешь? (например, RH%):")
value = input()
DATASET = pick_sensor(sensor_index)
print("выберите метод фильтрации: \n1 - скользящее среднее\n2 - экспоненциальное скользящее среднее\n3 - двойное экспоненциальное скользящее среднее\n4 - гауссов фильтр\n5 - фильтр Калмана\n6 - нефильтрованный сигнал\n7 - фильтр Фурье\n8 - фильтр Савицкого-Голая\n9 - завершить работу")
while True:
    method = int(input())
    match method:
        case 1:
            plot_moving_average(DATASET)
            finishing_touches(value)
        case 2:
            plot_exponential_moving_average(DATASET)
            finishing_touches(value)
        case 3:
            plot_double_exponential_moving_average(DATASET)
            finishing_touches(value)
        case 4:
            plot_gaussian_filter(DATASET)
            finishing_touches(value)
        case 5:
            plot_kalman_filter(DATASET)
            finishing_touches(value)
        case 6:
            plot_unfiltered(DATASET)
            finishing_touches(value)
        case 7:
            plot_fourier_filter(DATASET)
            finishing_touches(value)
        case 8:
            plot_savitzky_golay_filter(DATASET)
            finishing_touches(value)
        case 9:
            print("Работа завершена")
            break