import time
import numpy as np
from scipy.signal import welch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

def compute_alpha_power(data, fs=200):
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    alpha_power = np.sum(Pxx[alpha_idx])
    return alpha_power

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.mac_address = "f0:17:3b:41:ec:7d"
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    print("Streaming... press Ctrl+C to stop.")

    exg_channels = BoardShim.get_exg_channels(board_id)
    channel_of_interest = exg_channels[0]

    fs = BoardShim.get_sampling_rate(board_id)
    window_sec = 3
    window_size = int(window_sec * fs)

    threshold = 3.25e-5  # example threshold from offline analysis
    time.sleep(3)  # wait for data to accumulate
    try:
        while True:
            # Grab a large chunk from the ring buffer, e.g. 5000 samples
            data = board.get_current_board_data(5000)
            num_samples = data.shape[1]

            if num_samples >= window_size:
                segment = data[channel_of_interest, -window_size:]

                # Example: bandpass + notch, then Welch
                segment_c = np.ascontiguousarray(segment)
                # DataFilter.perform_bandpass(segment_c, fs, 3.0, 30.0, 4,
                #                             FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
                # DataFilter.perform_bandstop(segment_c, fs, 60.0, 4.0, 2,
                #                             FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)

                alpha_power = compute_alpha_power(segment_c, fs)
                eyes_closed = alpha_power > threshold
                print(f"AlphaPower={alpha_power:.6e} => Eyes {'CLOSED' if eyes_closed else 'OPEN'}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping real-time alpha detection...")

    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()
