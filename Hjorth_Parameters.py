import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read EEG data from Excel file using Pandas
dataframe = pd.read_excel('E:/Application/Research_Engineer_(UIU)/Data.xlsx')
eeg_data = dataframe['Signal'].values


# a. Plot the signal in time domain
plt.plot(eeg_data)
plt.ylim(4000, 4400)
plt.xlabel('Number of samples in 190 sec')
plt.ylabel('Amplitude')
plt.title('Signal in time domain')
plt.show()

# Create MNE RawArray object
sfreq = 128  # Sampling rate
info = mne.create_info(ch_names=['Signal'], sfreq=sfreq, ch_types=['eeg'])
raw = mne.io.RawArray(data=[eeg_data], info=info)

# a. Plot the signal in time domain
#raw.plot(duration=5, scalings='auto')

# a. Plot the signal in frequency domain
raw.plot_psd(fmax=64)  # Set fmax to limit the frequency range in the plot
plt.xlim(0, 70)
plt.ylim(80, 160)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Signal in frequency domain')

# b. Remove noise from the signal (optional, adapt to your specific noise removal method)
# Example: Apply a bandpass filter to remove noise between 1 Hz and 50 Hz
raw.filter(5, 45, fir_design='firwin')
raw.plot_psd(fmax=64)  
plt.xlim(0, 70)
plt.ylim(80, 160)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Filtered Signal in frequency domain')




# c. Make 5-second segments and plot the average of all segments
segment_length = 5 * sfreq
n_segments = len(eeg_data) // segment_length
segments = np.array_split(eeg_data, n_segments)
segment_means = np.mean(segments, axis=0)

plt.plot(segment_means)
plt.xlabel('Number of samples')
plt.ylabel('Amplitude')
plt.title('Average(mean) of 5-second segments')



###############################
###############################
import antropy as ant
allactivity=np.var(eeg_data)
allmobility, allcomplexity = np.round(ant.hjorth_params(eeg_data), 4)
###############################
###############################


# d. Calculate Hjorth parameters between odd and even segments
odd_segments = segments[::2]
even_segments = segments[1::2]

odd_activity = [np.var(segment) for segment in odd_segments]
odd_mobility = [np.sqrt(np.var(np.diff(segment)) / np.var(segment)) for segment in odd_segments]
odd_complexity = [np.sqrt(np.var(np.diff(np.diff(segment))) / np.var(np.diff(segment))) / odd_mobility[i] for i, segment in enumerate(odd_segments)]

even_activity = [np.var(segment) for segment in even_segments]
even_mobility = [np.sqrt(np.var(np.diff(segment)) / np.var(segment)) for segment in even_segments]
even_complexity = [np.sqrt(np.var(np.diff(np.diff(segment))) / np.var(np.diff(segment))) / even_mobility[i] for i, segment in enumerate(even_segments)]

# e. Calculate Hjorth parameters for all segments
activity = [np.var(segment) for segment in segments]
mobility = [np.sqrt(np.var(np.diff(segment)) / np.var(segment)) for segment in segments]
complexity = [np.sqrt(np.var(np.diff(np.diff(segment))) / np.var(np.diff(segment))) / mobility[i] for i, segment in enumerate(segments)]



# Print the calculated Hjorth parameters
print("Hjorth Parameters - Odd Segments:")
for i in range(len(odd_segments)):
    print("Segment", i+1)
    odd_activity[i]= odd_activity[i]+ odd_activity[i];
    print("Mobility:", odd_mobility[i])
    print("Complexity:", odd_complexity[i])
    print()

print("Hjorth Parameters - Even Segments:")
for i in range(len(even_segments)):
    print("Segment", i+1)
    print("Activity:", even_activity[i])
    print("Mobility:", even_mobility[i])
    print("Complexity:", even_complexity[i])
    print()

print("Hjorth Parameters - All Segments:")
for i in range(len(segments)):
    print("Segment", i+1)
    print("Activity:", activity[i])
    print("Mobility:", mobility[i])
    print("Complexity:", complexity[i])
    print()



import scipy.stats as stats

even_activity = 869.9763252
odd_activity = 1154.215673

t_stat_activity, p_value_activity = stats.ttest_ind([even_activity], [odd_activity])
print("Activity: t-statistic =", t_stat_activity, "p-value =", p_value_activity)



even_mobility = 0.263991189
odd_mobility = 0.316379981

t_stat_mobility, p_value_mobility = stats.ttest_ind([even_mobility], [odd_mobility])
print("Mobility: t-statistic =", t_stat_mobility, "p-value =", p_value_mobility)



even_complexity = 4.629867491
odd_complexity = 3.923706752

t_stat_complexity, p_value_complexity = stats.ttest_ind([even_complexity], [odd_complexity])
print("Complexity: t-statistic =", t_stat_complexity, "p-value =", p_value_complexity)






