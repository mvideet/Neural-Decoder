import time
import numpy as np
import pandas as pd
from scipy.signal import welch
import pyqtgraph as pg
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import winsound
from ML.realtime_predictor import RealTimePredictor


def beep():
    """Simple placeholder for beep sound."""
    print("(Beep!)")  # Replace with winsound.Beep(...) on Windows or an OS-specific beep.
    winsound.Beep(900, 750)


class EEGDataCollector:
    def __init__(self, mac_address="f0:17:3b:41:ec:7d"):
        """Initialize the EEG data collector with board setup"""
        # Set up BrainFlow board
        self.params = BrainFlowInputParams()
        self.params.mac_address = mac_address
        self.board_id = BoardIds.GANGLION_NATIVE_BOARD.value
        self.board = BoardShim(self.board_id, self.params)
        
        # Initialize board
        self.board.prepare_session()
        self.board.start_stream()
        print("Session prepared. Board streaming started.")
        
        # Get channel and sampling rate info
        self.channel = BoardShim.get_exg_channels(self.board_id)[0]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        
        # Initialize prediction window
        self.prediction_window = None
    
    def collect_segment(self, duration, label):
        """
        Collect 'duration' seconds of data from the board,
        return the raw signal and the label.
        """
        print(f"\n--- Recording {duration}s for label: {label} ---")
        start_time = time.time()
        recorded_data = []

        while (time.time() - start_time) < duration:
            # Fetch new data from the board
            data_chunk = self.board.get_board_data()
            channel_data = data_chunk[self.channel, :].tolist()
            recorded_data.extend(channel_data)
            time.sleep(0.1)  # Sleep briefly to avoid excessive CPU use

        # Convert to NumPy array
        recorded_data = np.array(recorded_data)
        
        print(f"Finished recording segment ({label}).")
        return recorded_data, label
    
    def run_continuous_prediction(self, predictor):
        """
        Continuously collect data and make predictions using a 600-point window.
        Every 1.5 seconds, moves points [300:600] to [0:300] and gets new points for [300:600].
        
        Args:
            predictor: Trained RealTimePredictor instance
        """
        print("\nStarting continuous prediction. Press Ctrl+C to stop.")
        window_size = 600
        update_size = 300
        update_interval = 1.5  # seconds between updates
        
        # Initialize the prediction window with first 600 points
        initial_data = self.board.get_current_board_data(window_size)[self.channel]
        if len(initial_data) < window_size:
            print("Waiting for enough initial data...")
            time.sleep(2)
            initial_data = self.board.get_current_board_data(window_size)[self.channel]
        
        self.prediction_window = initial_data
        
        try:
            while True:
                # Get most recent 300 points
                new_data = self.board.get_current_board_data(update_size)[self.channel]
                
                if len(new_data) >= update_size:
                    # Move second half of window to first half
                    self.prediction_window[:update_size] = self.prediction_window[update_size:]
                    # Put new data in second half
                    self.prediction_window[update_size:] = new_data
                    
                    # Make prediction on full 600-point window
                    prediction, confidence = predictor.predict_point(self.prediction_window)
                    state = "CLOSED" if prediction == 1 else "OPEN"
                    print(f"Eyes: {state}, Confidence: {confidence:.2f}")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nStopping continuous prediction.")
    
    def collect_training_data(self, segment_duration=30):
        """Collect training data with alternating eyes open/closed segments"""
        data_segments = []
        labels = []

        try:
            input("Press ENTER to begin the first segment (Eyes OPEN).")
            segment_data, seg_label = self.collect_segment(segment_duration, "open")
            data_segments.append(segment_data)
            labels.append(seg_label)
            beep()

            input("Press ENTER for second segment (Eyes CLOSED).")
            segment_data, seg_label = self.collect_segment(segment_duration, "closed")
            data_segments.append(segment_data)
            labels.append(seg_label)
            beep()

            input("Press ENTER for third segment (Eyes OPEN).")
            segment_data, seg_label = self.collect_segment(segment_duration, "open")
            data_segments.append(segment_data)
            labels.append(seg_label)
            beep()

            input("Press ENTER for fourth segment (Eyes CLOSED).")
            segment_data, seg_label = self.collect_segment(segment_duration, "closed")
            data_segments.append(segment_data)
            labels.append(seg_label)
            beep()

        except KeyboardInterrupt:
            print("Data collection interrupted by user.")
        finally:
            # Stop board streaming
            self.board.stop_stream()
            if self.board.is_prepared():
                self.board.release_session()

        # Combine all data
        combined_data = np.concatenate(data_segments)
        sample_labels = []
        for label, segment_data in zip(labels, data_segments):
            # Assign 1 for closed eyes, 0 for open eyes
            is_closed = 1 if label == "closed" else 0
            segment_labels = [is_closed] * len(segment_data)
            sample_labels.extend(segment_labels)
        
        # Create DataFrame and numpy array in the required format
        df = pd.DataFrame({'EXG Channel 0': combined_data})
        labels_array = np.array(sample_labels, dtype=int)
        
        print("\nData collection completed.")
        return df['EXG Channel 0'].values, labels_array
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'board'):
                if self.board.is_prepared():
                    self.board.stop_stream()
                    self.board.release_session()
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Collect training data
    collector = EEGDataCollector()
    channel_data, labels = collector.collect_training_data()
    
    # Create and train the predictor
    predictor = RealTimePredictor()
    print("\nTraining neural network...")
    predictor.train(channel_data, labels, epochs=20)
    
    # Start continuous prediction
    collector.run_continuous_prediction(predictor)
