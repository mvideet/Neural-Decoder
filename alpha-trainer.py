import time
import numpy as np
from scipy.signal import welch
import pyqtgraph as pg
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import winsound


def beep():
    """Simple placeholder for beep sound."""
    print("(Beep!)")  # Replace with winsound.Beep(...) on Windows or an OS-specific beep.
    winsound.Beep(900, 750)


def compute_alpha_power(signal, fs):
    """Compute alpha power (8-12 Hz) from an EEG segment using Welch's method."""
    from scipy.signal import welch
    f, Pxx = welch(signal, fs=fs, nperseg=fs // 2)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    return np.sum(Pxx[alpha_idx])


def collect_segment(board_shim, channel, duration, label):
    """
    Collect 'duration' seconds of data from 'channel' on 'board_shim',
    apply basic filtering, return the raw signal and the label.
    """
    fs = BoardShim.get_sampling_rate(board_shim.get_board_id())
    num_points_needed = duration * fs

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

    # Simple filters
    DataFilter.detrend(recorded_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(
        recorded_data, fs, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
    )
    DataFilter.perform_bandstop(
        recorded_data, fs, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
    )
    # Normalize data
    recorded_data = 2 * (recorded_data - np.min(recorded_data)) / (np.max(recorded_data) - np.min(recorded_data)) - 1

    print(f"Finished recording segment ({label}).")
    return recorded_data, label


def main():
    # -------------------------------------------------------------------------
    # 1) SET UP BRAINFLOW BOARD (SINGLE CHANNEL FOR DEMO)
    # -------------------------------------------------------------------------
    params = BrainFlowInputParams()
    params.mac_address = "f0:17:3b:41:ec:7d"  # Adjust for your device
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    board_shim = BoardShim(board_id, params)

    board_shim.prepare_session()
    board_shim.start_stream()
    print("Session prepared. Board streaming started.")

    # Channel of interest
    channel_of_interest = BoardShim.get_exg_channels(board_id)[0]
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # -------------------------------------------------------------------------
    # 2) COLLECT DATA IN LABELED SEGMENTS
    # -------------------------------------------------------------------------
    # We will do 2 cycles of (eyes open -> eyes closed) for demonstration
    # Each segment = 30 seconds
    # You can adapt for more or fewer cycles

    segment_duration = 15
    data_segments = []
    labels = []

    try:
        input("Press ENTER to begin the first segment (Eyes OPEN).")
        segment_data, seg_label = collect_segment(board_shim, channel_of_interest, segment_duration, "open")
        data_segments.append(segment_data)
        labels.append(seg_label)
        beep()

        input("Press ENTER for second segment (Eyes CLOSED).")
        segment_data, seg_label = collect_segment(board_shim, channel_of_interest, segment_duration, "closed")
        data_segments.append(segment_data)
        labels.append(seg_label)
        beep()

        input("Press ENTER for third segment (Eyes OPEN).")
        segment_data, seg_label = collect_segment(board_shim, channel_of_interest, segment_duration, "open")
        data_segments.append(segment_data)
        labels.append(seg_label)
        beep()

        input("Press ENTER for fourth segment (Eyes CLOSED).")
        segment_data, seg_label = collect_segment(board_shim, channel_of_interest, segment_duration, "closed")
        data_segments.append(segment_data)
        labels.append(seg_label)
        beep()

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")

    # Stop board streaming
    board_shim.stop_stream()
    if board_shim.is_prepared():
        board_shim.release_session()

    # -------------------------------------------------------------------------
    # 3) COMBINE ALL SEGMENTS & CREATE GROUND-TRUTH LABELS
    # -------------------------------------------------------------------------
    # We will produce one big array of EEG, plus an array labeling each sample
    # as “open” or “closed” eyes. Then we do alpha-power analysis in a sliding window.

    combined_data = np.concatenate(data_segments)
    sample_labels = []
    for label, segment_data in zip(labels, data_segments):
        # Assign "True" for closed eyes, "False" for open eyes, or any scheme you prefer
        is_closed = (label == "closed")
        segment_labels = [is_closed] * len(segment_data)
        sample_labels.extend(segment_labels)
    sample_labels = np.array(sample_labels, dtype=bool)

    print("\nData collection completed. Computing alpha power in a sliding window...")

    # -------------------------------------------------------------------------
    # 4) ALPHA POWER ANALYSIS (SLIDING WINDOW)
    # -------------------------------------------------------------------------
    window_size_sec = 3  # 3-second window
    step_sec = 1.5       # 50% overlap
    window_size = int(window_size_sec * sampling_rate)
    step_size = int(step_sec * sampling_rate)

    alpha_values = []
    time_values = []
    ground_truth = []  # Will store 'True' (eyes closed) or 'False' (eyes open) for each window

    idx = 0
    while (idx + window_size) <= len(combined_data):
        window_data = combined_data[idx:idx + window_size]
        alpha_val = compute_alpha_power(window_data, fs=sampling_rate)
        alpha_values.append(alpha_val)
        time_sec = idx / sampling_rate
        time_values.append(time_sec)

        # We consider the "majority label" in this window as ground truth
        window_labels = sample_labels[idx:idx + window_size]
        if np.mean(window_labels) > 0.5:
            ground_truth.append(True)   # Mostly closed
        else:
            ground_truth.append(False)  # Mostly open

        idx += step_size

    alpha_values = np.array(alpha_values)
    time_values = np.array(time_values)

    # -------------------------------------------------------------------------
    # 5) FIND OPTIMAL THRESHOLD TO MINIMIZE TOTAL ERROR
    # -------------------------------------------------------------------------
    # We try a range of thresholds. This is a simple example; you can refine.
    thresholds = np.linspace(alpha_values.min(), alpha_values.max(), 100) # 100 threshold values to check
    best_threshold = None
    best_error = float("inf")

    for thr in thresholds:
        # Predict closed if alpha > thr
        predictions = alpha_values > thr
        # Calculate false positives & false negatives
        # The XOR (^) operator identifies mismatches between predictions and ground_truth:
        # If predictions[i] matches ground_truth[i], XOR gives False (no error).
        # If they differ (one is True, the other is False), XOR gives True (error).
    
        mismatches = predictions ^ ground_truth  # XOR
        
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

    # -------------------------------------------------------------------------
    # 6) DISPLAY PYQTGRAPH OF ALPHA POWER vs. TIME, THRESHOLD, SEGMENT BOUNDARIES
    # -------------------------------------------------------------------------
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle("Alpha Power Training Results")
    plot = win.addPlot()
    plot.setLabel('bottom', 'Time (s)')
    plot.setLabel('left', 'Alpha Power')
    plot.setTitle("Alpha Power vs. Time - Training Segments")

    # Plot alpha power
    alpha_curve = plot.plot(time_values, alpha_values, pen='b', name="Alpha Power")

    # Plot threshold line
    threshold_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
    threshold_line.setPos(best_threshold)
    plot.addItem(threshold_line)

    # Draw vertical lines at segment boundaries (every 30 seconds)
    # We recorded 4 segments, each 30s. 
    # If you repeated more or less, adapt this logic as needed.
    boundary_times = [15, 30, 45, 60]  # where each 30s segment ended
    for bt in boundary_times:
        vline = pg.InfiniteLine(bt, angle=90, pen=pg.mkPen('g', style=pg.QtCore.Qt.DashLine))
        plot.addItem(vline)

    # Display final window with all plots
    win.show()
    print("\nClose the graph window to exit.")
    app.exec_()


if __name__ == "__main__":
    main()
