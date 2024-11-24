import numpy as np
import wfdb
import matplotlib.pyplot as plt

# Haar-like matched filter
def haar_like_matched_filter(ecg_signal, fs=360):
    """
    Apply a Haar-like matched filter to the ECG signal.
    
    Parameters:
    - ecg_signal: Input ECG signal (1D array).
    - fs: Sampling frequency (default: 360 Hz).
    
    Returns:
    - filtered_signal: Output of the Haar-like matched filter.
    """
    # Define B1 and B2 based on fs
    B1 = int(0.025 * fs)
    B2 = int(0.06 * fs)
    
    # Create Haar-like filter
    c = 2 * (B2 - B1) / (2 * B1 + 1)
    h = np.zeros(2 * B2 + 1)
    h[:B1] = c
    h[B1:B2] = -1
    h[-B2:] = -1
    h[-B1:] = c

    # Convolve the ECG signal with the Haar-like filter
    filtered_signal = np.convolve(ecg_signal, h, mode='same')
    return filtered_signal

# Second order difference
def second_order_difference(signal):
    """
    Compute the second order difference of a signal.
    
    Parameters:
    - signal: Input signal (1D array).
    
    Returns:
    - second_diff: Second order difference of the signal.
    """
    return np.diff(signal, n=2, prepend=0, append=0)

# Candidate sifting
def candidate_sifting(filtered_signal, ecg_signal, fs=360, c1=0.55):
    """
    Perform candidate sifting to identify potential R-wave peaks.
    
    Parameters:
    - filtered_signal: Output of the Haar-like matched filter.
    - ecg_signal: Original ECG signal.
    - fs: Sampling frequency (default: 360 Hz).
    - c1: Weight for second-order difference in score function.
    
    Returns:
    - candidates: Indices of candidate R-wave peaks.
    """
    # Compute second-order difference
    second_diff = second_order_difference(ecg_signal)
    
    # Compute the score function
    score = filtered_signal + c1 * second_diff
    
    # Candidate sifting
    candidates = []
    min_interval = int(0.2 * fs)  # Minimum interval between candidates
    for i in range(len(score)):
        window = score[max(0, i - min_interval):min(len(score), i + min_interval + 1)]
        if score[i] == np.max(window):
            candidates.append(i)
    
    return np.array(candidates)

# Adaptive thresholding
def adaptive_thresholding(candidates, scores, fs=360, T=0.1):
    """
    Perform adaptive thresholding to refine R-wave detection.
    
    Parameters:
    - candidates: Candidate indices.
    - scores: Score values at candidate indices.
    - fs: Sampling frequency (default: 360 Hz).
    - T: Constant for threshold adjustment.
    
    Returns:
    - refined_candidates: Indices of refined R-wave peaks.
    """
    adaptive_thresholds = []
    refined_candidates = []
    
    for i, candidate in enumerate(candidates):
        if i < 5:  # Not enough history for adaptive thresholding
            adaptive_thresholds.append(T)
            continue

        recent_scores = scores[max(0, i - 5):i]
        S5 = np.sort(np.abs(recent_scores))[-5]
        threshold = 5 * T + S5
        adaptive_thresholds.append(threshold)

        if scores[i] > threshold:
            refined_candidates.append(candidate)

    return np.array(refined_candidates)

# Variation ratio test
def variation_ratio_test(refined_candidates, ecg_signal, fs=360):
    """
    Apply the variation ratio test to remove noise-induced peaks.
    
    Parameters:
    - refined_candidates: Candidate indices after adaptive thresholding.
    - ecg_signal: Original ECG signal.
    - fs: Sampling frequency (default: 360 Hz).
    
    Returns:
    - final_peaks: Indices of final R-wave peaks.
    """
    final_peaks = []
    for candidate in refined_candidates:
        window = ecg_signal[max(0, candidate - int(0.1 * fs)):candidate + int(0.1 * fs)]
        u1 = np.max(window) - np.min(window)
        u2 = np.sum(np.abs(np.diff(window)))
        omega = u1 / u2 if u2 != 0 else 0

        if 0.4 <= omega <= 0.6:  # Threshold for variation ratio
            final_peaks.append(candidate)

    return np.array(final_peaks)

# Main function
def detect_r_peaks(ecg_signal, fs=360):
    """
    Detect R-wave peaks in an ECG signal.
    
    Parameters:
    - ecg_signal: Input ECG signal (1D array).
    - fs: Sampling frequency (default: 360 Hz).
    
    Returns:
    - r_peaks: Detected R-wave peaks.
    """
    # Step 1: Haar-like matched filter
    filtered_signal = haar_like_matched_filter(ecg_signal, fs)

    # Step 2: Candidate sifting
    candidates = candidate_sifting(filtered_signal, ecg_signal, fs)

    # Step 3: Adaptive thresholding
    scores = filtered_signal[candidates]
    refined_candidates = adaptive_thresholding(candidates, scores, fs)

    # Step 4: Variation ratio test
    r_peaks = variation_ratio_test(refined_candidates, ecg_signal, fs)
    
    return r_peaks


#read signal
database = "mit-bih-data"
record_num = "101"
sig = wfdb.rdsamp(database+ "/" + record_num)
x_both = sig[0][:].flatten()
x = x_both[0::2]
x2 = x_both[1::2]
#read sampling frequency
fs = wfdb.rdsamp(database + "/"+ record_num)[1]['fs']
#read annotation file
record_name = database+ "/" + record_num
annotation = wfdb.rdann(record_name, 'atr')
beat_peaks = annotation.sample[:]
#apply haar filter

detected = detect_r_peaks(x, fs)


#tolerance for beat detection of 5 samples
tolerance = 10

not_detected = []
for i in beat_peaks:
    for j in range(i-tolerance, i+tolerance+1):
        if j in detected :
            break
    if j == i+tolerance:
        not_detected.append(i)
        #print(i)


falsely_detected = []
for i in detected:
    for j in range(i-tolerance, i+tolerance+1):
        if j in beat_peaks:
            break
    if j == i+tolerance:
        falsely_detected.append(i)


print('Not detected:', len(not_detected))
print('Falsely detected:', len(falsely_detected))

SE = (len(detected) - len(falsely_detected))/len(beat_peaks)
PP = (len(detected) - len(falsely_detected))/len(detected) 
print('Sensitivity:', SE)
print('Positive predictivity:', PP)


start_int = 0
end_int = start_int + 5000
plt.plot(range(start_int, end_int), x[start_int:end_int])
detected_interval = [n for n in detected if n > start_int and n < end_int]
plt.plot(detected_interval, [x[n] for n in detected_interval], 'ro')
beat_peaks_interval = [n for n in beat_peaks if n > start_int and n < end_int]
plt.plot(beat_peaks_interval, [x[n] + 0.2 for n in beat_peaks_interval], 'bo')
plt.show()
