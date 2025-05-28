#!/usr/bin/env python3
"""
Neural-Decoder Training Script

Collects EEG data and trains the alpha wave detection model using both
traditional threshold-based and neural network approaches.

Usage:
    python scripts/train_model.py
"""

import sys
import os
import time
import numpy as np
from scipy.signal import welch
import pyqtgraph as pg
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import winsound

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_loader import config


def beep():
    """Simple placeholder for beep sound."""
    print("(Beep!)")
    try:
        winsound.Beep(900, 750)
    except:
        pass  # On non-Windows systems


def compute_alpha_power(signal, fs):
    """Compute alpha power (8-12 Hz) from an EEG segment using Welch's method."""
    alpha_config = config.get_signal_processing_config().get('alpha_band', {})
    low_freq = alpha_config.get('low_freq', 8.0)
    high_freq = alpha_config.get('high_freq', 12.0)
    
    f, Pxx = welch(signal, fs=fs, nperseg=fs // 2)
    alpha_idx = np.where((f >= low_freq) & (f <= high_freq))
    return np.sum(Pxx[alpha_idx])


def collect_segment(board_shim, channel, duration, label):
    """
    Collect 'duration' seconds of data from 'channel' on 'board_shim',
    apply basic filtering, return the raw signal and the label.
    """
    fs = BoardShim.get_sampling_rate(board_shim.get_board_id())
    
    print(f"\n--- Recording {duration}s for label: {label} ---")
    start_time = time.time()
    recorded_data = []

    while (time.time() - start_time) < duration:
        # Fetch new data from the board
        data_chunk = board_shim.get_board_data()  # gets all new data
        channel_data = data_chunk[channel, :].tolist()
        recorded_data.extend(channel_data)
        time.sleep(0.1)  # Sleep briefly to avoid excessive CPU use

    # Convert to NumPy array
    recorded_data = np.array(recorded_data)

    # Get signal processing configuration
    sig_config = config.get_signal_processing_config()
    bandpass_config = sig_config.get('bandpass', {})
    notch_config = sig_config.get('notch', {})

    # Apply filters using configuration
    DataFilter.detrend(recorded_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(
        recorded_data, fs, 
        bandpass_config.get('low_freq', 3.0), 
        bandpass_config.get('high_freq', 45.0), 
        bandpass_config.get('order', 2),
        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
    )
    DataFilter.perform_bandstop(
        recorded_data, fs, 
        notch_config.get('low_freq', 58.0), 
        notch_config.get('high_freq', 62.0), 
        notch_config.get('order', 2),
        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
    )
    
    # Normalize data
    recorded_data = 2 * (recorded_data - np.min(recorded_data)) / (np.max(recorded_data) - np.min(recorded_data)) - 1

    print(f"Finished recording segment ({label}).")
    return recorded_data, label


def main():
    """Main training function."""
    print("Neural-Decoder Training Script")
    print("=" * 40)
    
    # Load configuration
    device_config = config.get_device_config()
    data_config = config.get('data_collection', {})
    sig_config = config.get_signal_processing_config()
    
    # Set up BrainFlow board
    params = BrainFlowInputParams()
    params.mac_address = device_config.get('mac_address', 'f0:17:3b:41:ec:7d')
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    board_shim = BoardShim(board_id, params)

    board_shim.prepare_session()
    board_shim.start_stream()
    print("Session prepared. Board streaming started.")

    # Channel of interest
    channel_of_interest = BoardShim.get_exg_channels(board_id)[0]
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # Data collection parameters
    segment_duration = data_config.get('segment_duration', 15)
    labels_sequence = data_config.get('labels', ['open', 'closed', 'open', 'closed'])
    
    data_segments = []
    labels = []

    try:
        for i, label in enumerate(labels_sequence, 1):
            input(f"Press ENTER to begin segment {i}/4 (Eyes {label.upper()}).")
            segment_data, seg_label = collect_segment(
                board_shim, channel_of_interest, segment_duration, label
            )
            data_segments.append(segment_data)
            labels.append(seg_label)
            beep()

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")

    # Stop board streaming
    board_shim.stop_stream()
    if board_shim.is_prepared():
        board_shim.release_session()

    # Combine all segments & create ground-truth labels
    combined_data = np.concatenate(data_segments)
    sample_labels = []
    for label, segment_data in zip(labels, data_segments):
        is_closed = (label == "closed")
        segment_labels = [is_closed] * len(segment_data)
        sample_labels.extend(segment_labels)
    sample_labels = np.array(sample_labels, dtype=bool)

    print("\nData collection completed. Computing alpha power in a sliding window...")

    # Alpha power analysis (sliding window)
    window_size_sec = sig_config.get('window_size_sec', 3.0)
    overlap_percent = sig_config.get('overlap_percent', 0.5)
    step_sec = window_size_sec * overlap_percent
    
    window_size = int(window_size_sec * sampling_rate)
    step_size = int(step_sec * sampling_rate)

    alpha_values = []
    time_values = []
    ground_truth = []

    idx = 0
    while (idx + window_size) <= len(combined_data):
        window_data = combined_data[idx:idx + window_size]
        alpha_val = compute_alpha_power(window_data, fs=sampling_rate)
        alpha_values.append(alpha_val)
        time_sec = idx / sampling_rate
        time_values.append(time_sec)

        # Majority label in this window as ground truth
        window_labels = sample_labels[idx:idx + window_size]
        if np.mean(window_labels) > 0.5:
            ground_truth.append(True)   # Mostly closed
        else:
            ground_truth.append(False)  # Mostly open

        idx += step_size

    alpha_values = np.array(alpha_values)
    time_values = np.array(time_values)

    # Find optimal threshold to minimize total error
    thresholds = np.linspace(alpha_values.min(), alpha_values.max(), 100)
    best_threshold = None
    best_error = float("inf")

    for thr in thresholds:
        predictions = alpha_values > thr
        mismatches = predictions ^ ground_truth
        total_error = np.sum(mismatches)
        if total_error < best_error:
            best_error = total_error
            best_threshold = thr

    # Compute final predictions & error rates
    final_predictions = alpha_values > best_threshold
    mismatches = final_predictions ^ ground_truth
    total_error_rate = np.mean(mismatches)
    false_positives = np.sum((final_predictions == True) & (ground_truth == False))
    false_negatives = np.sum((final_predictions == False) & (ground_truth == True))

    print(f"\nOptimal threshold found: {best_threshold:.6f}")
    print(f"Total error: {best_error} windows ({100.0 * total_error_rate:.2f}% of windows)")
    print(f"False Positives: {false_positives}, False Negatives: {false_negatives}")

    # Save results to data directory
    import pandas as pd
    results_df = pd.DataFrame({
        'time': time_values,
        'alpha_power': alpha_values,
        'ground_truth': ground_truth,
        'predictions': final_predictions
    })
    
    output_path = os.path.join('data', 'processed', 'training_results.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Display visualization
    vis_config = config.get_visualization_config()
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle(vis_config.get('window_title', 'Neural-Decoder Training Results'))
    plot = win.addPlot()
    plot.setLabel('bottom', 'Time (s)')
    plot.setLabel('left', 'Alpha Power')
    plot.setTitle("Alpha Power vs. Time - Training Segments")

    # Plot alpha power
    plot.plot(time_values, alpha_values, pen='b', name="Alpha Power")

    # Plot threshold line
    threshold_line = pg.InfiniteLine(angle=0, movable=False, 
                                   pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
    threshold_line.setPos(best_threshold)
    plot.addItem(threshold_line)

    # Draw vertical lines at segment boundaries
    boundary_times = [segment_duration * i for i in range(1, len(labels_sequence))]
    for bt in boundary_times:
        vline = pg.InfiniteLine(bt, angle=90, 
                              pen=pg.mkPen('g', style=pg.QtCore.Qt.DashLine))
        plot.addItem(vline)

    win.show()
    print("\nClose the graph window to exit.")
    app.exec_()


if __name__ == "__main__":
    main()
