import numpy as np
import matplotlib.pyplot as plt
from fathon import fathonUtils as fu
import fathon
from scipy.stats import norm
import colorednoise as cn
from sklearn.preprocessing import QuantileTransformer


def dfa(data, scales, order=1, plot=True):
    """Perform Detrended Fluctuation Analysis on data

    Inputs:
        data: 1D numpy array of time series to be analyzed.
        scales: List or array of scales to calculate fluctuations
        order: Integer of polynomial fit (default=1 for linear)
        plot: Return loglog plot (default=True to return plot)

    Outputs:
        scales: The scales that were entered as input
        fluctuations: Variability measured at each scale with RMS
        alpha value: Value quantifying the relationship between the scales
                     and fluctuations

....References:
........Damouras, S., Chang, M. D., Sejdi, E., & Chau, T. (2010). An empirical
..........examination of detrended fluctuation analysis for gait data. Gait &
..........posture, 31(3), 336-340.
........Mirzayof, D., & Ashkenazy, Y. (2010). Preservation of long range
..........temporal correlations under extreme random dilution. Physica A:
..........Statistical Mechanics and its Applications, 389(24), 5573-5580.
........Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
..........Quantification of scaling exponents and crossover phenomena in
..........nonstationary heartbeat time series. Chaos: An Interdisciplinary
..........Journal of Nonlinear Science, 5(1), 82-87.
# =============================================================================
                            ------ EXAMPLE ------

      - Generate random data
      data = np.random.randn(5000)

      - Create a vector of the scales you want to use
      scales = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]

      - Set a detrending order. Use 1 for a linear detrend.
      order = 1

      - run dfa function
      s, f, a = dfa(data, scales, order, plot=True)
# =============================================================================
"""

    # Check if data is a column vector (2D array with one column)
    if data.shape[0] == 1:
        # Reshape the data to be a column vector
        data = data.reshape(-1, 1)
    else:
        # Data is already a column vector
        data = data

    # =============================================================================
    ##########################   START DFA CALCULATION   ##########################
    # =============================================================================

    # Step 1: Integrate the data
    integrated_data = np.cumsum(data - np.mean(data))

    fluctuation = []

    for scale in scales:
        # Step 2: Divide data into non-overlapping window of size 'scale'
        chunks = len(data) // scale
        ms = 0.0

        for i in range(chunks):
            this_chunk = integrated_data[i * scale:(i + 1) * scale]
            x = np.arange(len(this_chunk))

            # Step 3: Fit polynomial (default is linear, i.e., order=1)
            coeffs = np.polyfit(x, this_chunk, order)
            fit = np.polyval(coeffs, x)

            # Detrend and calculate RMS for the current window
            ms += np.mean((this_chunk - fit) ** 2)

            # Calculate average RMS for this scale
        fluctuation.append(np.sqrt(ms / chunks))

        # Perform linear regression
    alpha, intercept = np.polyfit(np.log(scales), np.log(fluctuation), 1)

    # Create a log-log plot to visualize the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(scales, fluctuation, marker='o', markerfacecolor='red', markersize=8,
                   linestyle='-', color='black', linewidth=1.7, label=f'Alpha = {alpha:.3f}')
        plt.xlabel('Scale (log)')
        plt.ylabel('Fluctuation (log)')
        plt.legend()
        plt.title('Detrended Fluctuation Analysis')
        plt.grid(True)
        plt.show()

    # Return the scales used, fluctuation functions and the alpha value
    return scales, fluctuation, alpha

def DFA(variable):
    a = fu.toAggregated(variable)
        #b = fu.toAggregated(b)

    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(start=9, end=int(len(variable)/16))
    revSeg = True
    polOrd = 1

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept = pydfa.fitFlucVec()
    plt.plot(np.log(n), np.log(F), 'ro')
    plt.plot(np.log(n), H_intercept + H * np.log(n), 'k-', label='H = {:.2f}'.format(H))
    plt.xlabel('ln(n)', fontsize=14)
    plt.ylabel('ln(F(n))', fontsize=14)
    plt.title('DFA', fontsize=14)
    plt.legend(loc=0, fontsize=14)
    #plt.clf()
    plt.show()
    return H

def fgn_sim(n=1000, H=1):
    """Create Fractional Gaussian Noise
     Inputs:
            n: Number of data points of the time series. Default is 1000 data points.
            H: Hurst parameter of the time series. Default is 0.7.
     Outputs:
            An array of n data points with variability H
    # =============================================================================
                                ------ EXAMPLE ------

          - Create time series of 1000 datapoints to have an H of 0.7
          n = 1000
          H = 0.7
          dat = fgn_sim(n, H)

          - If you would like to plot the timeseries:
          import matplotlib.pyplot as plt
          plt.plot(dat)
          plt.title(f"Fractional Gaussian Noise (H = {H})")
          plt.xlabel("Time")
          plt.ylabel("Value")
          plt.show()
    # =============================================================================
    """

    # Settings:
    mean = 0
    std = 1

    # Generate Sequence:
    z = np.random.normal(size=2 * n)
    zr = z[:n]
    zi = z[n:]
    zic = -zi
    zi[0] = 0
    zr[0] = zr[0] * np.sqrt(2)
    zi[n - 1] = 0
    zr[n - 1] = zr[n - 1] * np.sqrt(2)
    zr = np.concatenate([zr[:n], zr[n - 2::-1]])
    zi = np.concatenate([zi[:n], zic[n - 2::-1]])
    z = zr + 1j * zi

    k = np.arange(n)
    gammak = (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H)) / 2
    ind = np.concatenate([np.arange(n - 1), [n - 1], np.arange(n - 2, 0, -1)])
    gammak = gammak[ind]  # Circular shift of gammak to match n
    gkFGN0 = np.fft.ifft(gammak)
    gksqrt = np.real(gkFGN0)

    if np.all(gksqrt > 0):
        gksqrt = np.sqrt(gksqrt)
        z = z[:len(gksqrt)] * gksqrt
        z = np.fft.ifft(z)
        z = 0.5 * (n - 1) ** (-0.5) * z
        z = np.real(z[:n])
    else:
        gksqrt = np.zeros_like(gksqrt)
        raise ValueError("Re(gk)-vector not positive")

    # Standardize: (z - np.mean(z)) / np.sqrt(np.var(z))
    ans = std * z + mean
    return ans

def Perc(signal , upper_lim, lower_lim):
    """This function takes a signal as a np.array and turns it as values from upper_lim to lower_lim"""
    if np.min(signal) < 0:
        signal = signal - np.min(signal)
    signal = 100 * signal / np.max(signal)
    min_val = signal.min()
    max_val = signal.max()
    signal = (signal - min_val) / (max_val - min_val)
    new_range = upper_lim - lower_lim
    signal = signal * new_range + lower_lim
    return signal
# signal1 = cn.powerlaw_psd_gaussian(1, 1000)
# signal1 = fgn_sim(n=100000,H=0.99)
# DFA = DFA(signal1)
#
# a = np.arange(4, int(len(signal1)/4))
# b = fu.linRangeByStep(start=4, end=int(len(signal1)/4))
# print(a)
# print(b)
# dfa = dfa(signal1, b ,1 ,True)
# print(dfa[2])
# print(DFA)

# result = []
#
# for i in range(50, 0, -1):
#     # Add 'i' to the list 'i' times
#     result.extend([i] * i)
#
#     # Add '100 - i + 1' to the list 'i' times
#     result.extend([101 - i] * i)
# def count_occurrences(num_list):
#     # Initialize two lists
#     numbers = list(range(1, 101))  # List from 1 to 100
#     counts = [0] * 100  # List to store counts of each number, initialized to 0
#
#     # Count occurrences of each number in num_list
#     for num in num_list:
#         counts[num - 1] += 1  # Increment the count at the index corresponding to the number
#
#     return numbers, counts
# numbers, counts = count_occurrences(result)
# plt.plot(numbers,counts)
# plt.show()

# Parameters for the normal distribution
# mean = 50     # Mean of the distribution
# std_dev = 25  # Standard deviation of the distribution
# size = 1000  # Number of data points
#
# # Generate the data series
# data_series = np.random.normal(mean, std_dev, size)
# time = np.arange(0,len(data_series))
# data_series_0_100 = Perc(data_series,100,0)
# # Print the first few values of the data series
#
#
# plt.scatter(time, data_series_0_100, label='data_series_0_100')
# plt.scatter(time, data_series, label='data_series')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.hist(data_series_0_100, bins=30, density=True, alpha=0.6, color='g')
#
# # Add a line showing the expected normal distribution
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
# plt.plot(x, p, 'k', linewidth=2)
#
# # Add titles and labels
# plt.title("Histogram of Normally Distributed Data")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
#
# # Show the plot
# plt.show()
# Function to adjust the maximum density
def adjust_max_density(data, desired_max_density):
    # Normalize the data to have maximum density of 1
    density, edges = np.histogram(data, bins=30, density=True)
    max_density = density.max()

    if max_density > 0:
        # Scale density to achieve the desired maximum density
        scaling_factor = desired_max_density / max_density
        data_adjusted = data * scaling_factor
    else:
        data_adjusted = data

    return data_adjusted

def generate_pink_noise(size, mean, std_dev):
    # Generate white noise
    white_noise = np.random.normal(mean, std_dev, size)

    # Perform the Fourier transform
    f = np.fft.rfft(white_noise)

    # Create frequencies corresponding to the FFT components
    frequencies = np.fft.rfftfreq(size)

    # Avoid division by zero for the DC component
    frequencies[0] = 1  # Set the DC component frequency to 1 to avoid division by zero

    # Scale the FFT components by 1/f (creating the pink noise characteristic)
    pink_spectrum = f / np.sqrt(frequencies)

    # Perform the inverse Fourier transform
    pink_noise = np.fft.irfft(pink_spectrum, n=size)

    # Normalize the pink noise to have a standard deviation of 1
    pink_noise /= np.std(pink_noise)

    pink_noise = Perc(pink_noise, 100, 0)

    return pink_noise

# pink_signal = generate_pink_noise(size=1000, mean=0, std_dev=1)
# print(pink_signal)
# plt.plot(pink_signal)
# plt.show()
# dfa = DFA(pink_signal)
# print(np.mean(pink_signal))
# # Plot the histogram (distribution) of the pink noise signal
# plt.figure(figsize=(8, 6))
# plt.hist(pink_signal, bins=15, density=True, alpha=0.6, color='magenta')
#
# # Add titles and labels
# plt.title("Distribution of Pink Noise Signal")
# plt.xlabel("Signal Amplitude")
# plt.ylabel("Density")
#
# # Show the plot
# plt.show()
#
# print(dfa)


def shift_density(data, new_peak, range_min=0, range_max=100):
    # Parameters of the original distribution
    original_mean = np.mean(data)
    original_std = np.std(data)

    # Shift and scale the data
    data_centered = data - original_mean  # Center the data around 0
    data_scaled = data_centered / original_std  # Scale data to unit variance

    # Apply the new peak shift
    new_mean = new_peak
    data_shifted = data_scaled * original_std + new_mean  # Rescale to original std deviation
    data_shifted = np.clip(data_shifted, range_min, range_max)  # Clip to the desired range

    return data_shifted

# Parameters for the original distribution

# data_original = generate_pink_noise(size=1000, mean=0, std_dev=1)
mean_original = 50
std_dev = 10
data_original = np.random.normal(loc=mean_original, scale=std_dev, size=1000)
data_original = Perc(data_original, 100, 0)

# Shift density peak to the new value
new_peak = 30
data_shifted = shift_density(data_original, new_peak)

# Plot the distributions
plt.figure(figsize=(12, 6))

# Original distribution
plt.subplot(1, 2, 1)
plt.hist(data_original, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_original, std_dev)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Original Distribution')
plt.xlabel('Value')
plt.ylabel('Density')

# Shifted distribution
plt.subplot(1, 2, 2)
plt.hist(data_shifted, bins=30, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
# Compute new mean and std_dev for visualization
new_mean = new_peak
new_std_dev = std_dev  # Keep the same standard deviation
p = norm.pdf(x, new_mean, new_std_dev)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Shifted Distribution')
plt.xlabel('Value')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
