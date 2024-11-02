import wfdb
from wfdb import processing
import pandas as pd
import numpy as np
from datetime import timedelta

import requests
from bs4 import BeautifulSoup
from scipy.signal import welch

import time

import boto3


# Use the signal for filters
from scipy.signal import butter, filtfilt, iirnotch


# Base path to download the patient overview
base_url = "https://physionet.org/content/i-care/2.1/training/"
records_url = base_url + "RECORDS"


# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = "ecg-data-ttm"
result_file_key = "result_Rest.txt"


# Function to log timing results to S3
def log_time_to_s3(patient_id, elapsed_time):
    try:
        # Create the log entry
        log_entry = f"Patient {patient_id} loading time: {elapsed_time:.2f} seconds\n"
        
        # Write to a temporary local file
        with open("/tmp/results.txt", "a") as file:
            file.write(log_entry)
        
        # Upload the results.txt file to S3
        s3.upload_file("/tmp/results.txt", bucket_name, result_file_key)
        print("Timing result uploaded to S3.")
    except Exception as e:
        print(f"Error uploading timing result to S3: {e}")

# Base directory path for the PhysioNet database
physionet_db_path = "i-care/2.1/training"

patient_numbers = []

# Download the RECORDS file to get the list of patient numbers
response = requests.get(records_url)
if response.status_code == 200:
    # The RECORDS file is a plain text file, so split it by lines to get the patient numbers
    patient_numbers_html = response.text
else:
    print("Failed to download the RECORDS file.")

# Parse the HTML content
soup = BeautifulSoup(patient_numbers_html, 'html.parser')

# Find the <pre> tag with class "plain" and the <code> tag within it
code_tag = soup.find('pre', {'class': 'plain'}).find('code')

# Extract the content of the <code> tag
patient_records = code_tag.text if code_tag else ''

# Split the content into individual records (one per line)
patient_records_list = patient_records.splitlines()

# Clean the list by removing any empty strings
patient_records_list = [record.strip() for record in patient_records_list if record.strip()]

# Clean up the patient numbers by removing 'training/' and the trailing '/'
patient_numbers = [record.split('/')[1] for record in patient_records_list]

# Print the cleaned patient numbers to verify
print(patient_numbers)

# Define the total number of segments based on `hours_included` in 5-minute intervals
hours_included = 24
total_segments = (hours_included * 60) // 5

# Define the columns by including `Mean_HR`, `HRV_SDNN`, `LF_Power`, `HF_Power`, and `LF/HF_Ratio` for each segment
columns = ["Patient_ID"] + [
    f"Segment_{i+1}_Mean_HR" if j == 0 else 
    f"Segment_{i+1}_HRV_SDNN" if j == 1 else 
    f"Segment_{i+1}_LF_Power" if j == 2 else 
    f"Segment_{i+1}_HF_Power" if j == 3 else 
    f"Segment_{i+1}_LF_HF_Ratio"
    for i in range(total_segments) for j in range(5)
]

# Create the DataFrame with specified columns
final_df = pd.DataFrame(columns=columns)

def apply_high_low_filter(signal, sampling_rate, utility_freq):
    # High-pass filter to remove baseline drift (cutoff frequency = 0.5 Hz, order = 2)
    b_high, a_high = butter(2, 0.5 / (sampling_rate / 2), btype='high')
    filtered_signal = filtfilt(b_high, a_high, signal)

    # Low-pass filter to remove high-frequency noise (cutoff frequency = 40 Hz, order = 2)
    b_low, a_low = butter(2, 40 / (sampling_rate / 2), btype='low')
    filtered_signal = filtfilt(b_low, a_low, filtered_signal)

    # Notch filter to remove utility frequency noise (e.g., 50 Hz or 60 Hz)
    Q = 30.0  # Quality factor for notch filter
    b_notch, a_notch = iirnotch(utility_freq, Q, sampling_rate)
    filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)

    return filtered_signal

def calculate_hr_hrv_lf_hf_ratio(signal, sampling_rate, local_start_time, total_samples):
    # Set thresholds and frequency bands
    hr_threshold = (30, 200)
    hrv_threshold = (0, 300)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # Calculate the duration to the next full 5-minute boundary from local_start_time
    first_segment_duration = timedelta(minutes=5) - timedelta(seconds=local_start_time.total_seconds() % 300)
    first_segment_duration_seconds = first_segment_duration.total_seconds()

    # Calculate the sample index to end the first segment
    first_segment_end_index = int(first_segment_duration_seconds * sampling_rate)
    first_segment_signal = signal[:first_segment_end_index]

    # Initialize XQRS for the first segment and detect QRS complexes
    xqrs = processing.XQRS(sig=first_segment_signal, fs=sampling_rate)
    xqrs.detect(learn=False, verbose=False)
    qrs_indices = np.array(xqrs.qrs_inds)  # Indices of detected R-peaks in sample points
    r_wave_times = qrs_indices / sampling_rate

    # Initialize lists to store HR, HRV, LF, HF values, LF/HF ratios, and QRS indices
    hr_values, hrv_values, lf_powers, hf_powers, lfhf_ratios = [], [], [], [], []
    all_qrs_indices = []  # List to accumulate QRS indices from each segment
    segment_start_indices = [0]

    # Append QRS indices for the first segment
    all_qrs_indices.extend(qrs_indices)

    # Process the shortened first segment
    first_segment_indices = r_wave_times < first_segment_duration_seconds
    first_segment_rr_intervals = np.diff(r_wave_times[first_segment_indices]) * 1000  # Convert to milliseconds

    # Calculate HR, HRV, LF, HF, and LF/HF for the first segment
    if len(first_segment_rr_intervals) > 0:
        mean_hr = int(round(processing.calc_mean_hr(first_segment_rr_intervals / 1000, rr_units="seconds")))
        hrv = int(round(np.std(first_segment_rr_intervals)))

        # Apply threshold conditions
        mean_hr = mean_hr if hr_threshold[0] <= mean_hr <= hr_threshold[1] else np.nan
        hrv = hrv if hrv_threshold[0] <= hrv <= hrv_threshold[1] else np.nan

        # Calculate LF and HF power
        freqs, psd = welch(first_segment_rr_intervals, fs=1.0, nperseg=len(first_segment_rr_intervals))
        lf_power = int(np.trapezoid(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])]))
        hf_power = int(np.trapezoid(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])]))
        lfhf_ratio = round((lf_power / hf_power), 2) if hf_power > 0 else np.nan
    else:
        mean_hr, hrv, lf_power, hf_power, lfhf_ratio = np.nan, np.nan, np.nan, np.nan, np.nan

    # Append results for the first segment
    hr_values.append(mean_hr)
    hrv_values.append(hrv)
    lf_powers.append(lf_power)
    hf_powers.append(hf_power)
    lfhf_ratios.append(lfhf_ratio)

    # Process remaining full 5-minute segments
    current_time = first_segment_duration_seconds
    num_segments = int((total_samples / sampling_rate - current_time) // 300)  # Remaining time for full segments
    for _ in range(num_segments):
        segment_start_index = int(current_time * sampling_rate)
        segment_start_indices.append(segment_start_index)
        
        segment_end = current_time + 300
        segment_signal = signal[segment_start_index:int(segment_end * sampling_rate)]
        
        # Initialize XQRS for the current segment
        xqrs = processing.XQRS(sig=segment_signal, fs=sampling_rate)
        xqrs.detect(learn=False, verbose=False)
        qrs_indices = np.array(xqrs.qrs_inds)
        all_qrs_indices.extend(qrs_indices + segment_start_index)  # Adjust indices to the full signal context
        selected_r_wave_times = (qrs_indices + segment_start_index) / sampling_rate  # Convert to seconds with offset
        segment_rr_intervals = np.diff(selected_r_wave_times) * 1000  # Convert RR intervals to milliseconds

        # Calculate HR, HRV, LF, HF, and LF/HF for the segment
        if len(segment_rr_intervals) > 0:
            mean_hr = int(round(processing.calc_mean_hr(segment_rr_intervals / 1000, rr_units="seconds")))
            hrv = int(round(np.std(segment_rr_intervals)))

            mean_hr = mean_hr if hr_threshold[0] <= mean_hr <= hr_threshold[1] else np.nan
            hrv = hrv if hrv_threshold[0] <= hrv <= hrv_threshold[1] else np.nan

            freqs, psd = welch(segment_rr_intervals, fs=1.0, nperseg=len(segment_rr_intervals))
            lf_power = int(np.trapezoid(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])]))
            hf_power = int(np.trapezoid(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])]))
            lfhf_ratio = round((lf_power / hf_power), 2) if hf_power > 0 else np.nan
        else:
            mean_hr, hrv, lf_power, hf_power, lfhf_ratio = np.nan, np.nan, np.nan, np.nan, np.nan

        # Append the results for each segment
        hr_values.append(mean_hr)
        hrv_values.append(hrv)
        lf_powers.append(lf_power)
        hf_powers.append(hf_power)
        lfhf_ratios.append(lfhf_ratio)

        # Move to the next 5-minute segment
        current_time = segment_end

    # Convert all_qrs_indices to a numpy array if needed
    all_qrs_indices = np.array(all_qrs_indices)

    return hr_values, hrv_values, lf_powers, hf_powers, lfhf_ratios, all_qrs_indices, segment_start_indices

def transfer_to_final_structure(patient_id, hr_hrv_df, final_df):
    # Create an empty row with NaN values for all segments
    row_data = {"Patient_ID": patient_id}
    for column in final_df.columns:
        if column != "Patient_ID":
            row_data[column] = np.nan

    # If hr_hrv_df has data, fill in the corresponding HR, HRV, LF, HF, and LF/HF values
    if not hr_hrv_df.empty:
        for i in range(len(hr_hrv_df)):
            hr_column = f"Segment_{i+1}_Mean_HR"
            hrv_column = f"Segment_{i+1}_HRV_SDNN"
            lf_column = f"Segment_{i+1}_LF_Power"
            hf_column = f"Segment_{i+1}_HF_Power"
            lfhf_ratio_column = f"Segment_{i+1}_LF_HF_Ratio"
            
            # Assign values using column names that match hr_hrv_df
            row_data[hr_column] = hr_hrv_df.at[i, "Mean_HR"]
            row_data[hrv_column] = hr_hrv_df.at[i, "HRV_SDNN"]
            row_data[lf_column] = hr_hrv_df.at[i, "LF_Power"]
            row_data[hf_column] = hr_hrv_df.at[i, "HF_Power"]
            row_data[lfhf_ratio_column] = hr_hrv_df.at[i, "LF_HF_Ratio"]

    # Convert row_data dictionary to a DataFrame and concatenate with final_df
    new_row_df = pd.DataFrame([row_data])
    final_df = pd.concat([final_df, new_row_df], ignore_index=True)
    return final_df

def get_patient_records(patient_number):
    # Construct the full URL to the RECORDS file
    file_url = f"{base_url}/{patient_number}/RECORDS"

    # Fetch the URL content
    response = requests.get(file_url)
    if response.status_code == 200:
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text content of the RECORDS file
        patient_records = soup.text  # Plain text response for RECORDS

        # Split the content into lines and filter for `_ECG` files with last 3 digits <= 023
        ecg_files = [
            line for line in patient_records.splitlines()
            if line.endswith("_ECG") and int(line.split("_")[-2]) <= (hours_included - 1)
        ]

        return ecg_files
    else:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")

# Start timer
start_timer = time.time()

# Get the first 20 entries
first_20_patients = patient_numbers[500:]

total_patients = len(first_20_patients)

for i, patient_number in enumerate(first_20_patients, start=1):
    print("===================================================")
    print(f"Starting Patient {i} of {total_patients} (Patient number {patient_number})")
    print("===================================================")

    

    ecg_files = get_patient_records(patient_number)


    # Initialize a DataFrame for a full 24-hour period, with 5-minute segments
    total_duration_hours = hours_included
    segment_duration = timedelta(minutes=5)
    total_segments = (total_duration_hours * 60) // 5

    # Create the full DataFrame with NaN values initially
    hr_hrv_df = pd.DataFrame({
        "Start Time": [timedelta(minutes=5 * i) for i in range(total_segments)],
        "Mean_HR": [np.nan] * total_segments,
        "HRV_SDNN": [np.nan] * total_segments,
        "LF_Power": [np.nan] * total_segments,
        "HF_Power": [np.nan] * total_segments,
        "LF_HF_Ratio": [np.nan] * total_segments
    })

    for ecg_file in ecg_files:

        # Construct the file's record name (without extension)
        record_name = ecg_file
        
        try:
            # Load the record using PhysioNet-specific directory
            record = wfdb.rdrecord(record_name, pn_dir=f"{physionet_db_path}/{patient_number}")
            print(f"Successfully loaded: {ecg_file}")
        except Exception as e:
            print(f"Failed to load {ecg_file}: {e}")

        # Get metadata of the signal
        patient_number = record.record_name.split("_")[0]
        sampling_rate = record.fs
        signal_len = record.sig_len
        utility_freq = int(record.comments[0].split(": ")[1])
        start_time = pd.to_timedelta(record.comments[1].split(": ")[1])
        end_time = pd.to_timedelta(record.comments[2].split(": ")[1])

        # Access the signal data (numpy array) for the second lead
        if record.p_signal.shape[1] > 1:  # Check if there are multiple leads
            ecg_signal_data = record.p_signal[:, 0].flatten()  # Select the second lead
        else:
            # Access the signal data (numpy array)
            ecg_signal_data = record.p_signal.flatten()

        filtered_signal = apply_high_low_filter(ecg_signal_data, sampling_rate, utility_freq)
        # Call the updated function to calculate HR, HRV, LF, HF, and LF/HF ratio
        hr_values, hrv_values, lf_values, hf_values, lfhf_ratios, qrs_indices, segment_start_points = calculate_hr_hrv_lf_hf_ratio(
            filtered_signal, sampling_rate, start_time, signal_len
        )
        print(f"hr_values: {hr_values}")
        print(f"hrv_values: {hrv_values}")
        print(f"lf_values: {lf_values}")
        print(f"hf_values: {hf_values}")
        print(f"lfhf_ratios: {lfhf_ratios}")
        # Find the nearest 5-minute interval for `start_time`
        start_index = hr_hrv_df[hr_hrv_df["Start Time"] <= start_time].last_valid_index()

        # Populate `hr_hrv_df` starting from `start_index`
        for i, (hr, hrv, lf, hf, lfhf) in enumerate(zip(hr_values, hrv_values, lf_values, hf_values, lfhf_ratios)):
            if start_index + i < len(hr_hrv_df):  # Ensure we don't go out of bounds
                hr_hrv_df.at[start_index + i, "Mean_HR"] = hr
                hr_hrv_df.at[start_index + i, "HRV_SDNN"] = hrv
                hr_hrv_df.at[start_index + i, "LF_Power"] = lf
                hr_hrv_df.at[start_index + i, "HF_Power"] = hf
                hr_hrv_df.at[start_index + i, "LF_HF_Ratio"] = lfhf

    # After processing all files for a patient, call this function
    final_df = transfer_to_final_structure(patient_number, hr_hrv_df, final_df)

# After processing, stop the timer and calculate elapsed time
end_timer = time.time()
elapsed_time = end_timer - start_timer

final_df.to_csv('/tmp/hr_hrv_df.csv', index=False)
s3.upload_file('/tmp/hr_hrv_df.csv', 'ecg-data-ttm', 'hr_hrv_df_Rest.csv')

# Log the timing to S3
log_time_to_s3(patient_number, elapsed_time)