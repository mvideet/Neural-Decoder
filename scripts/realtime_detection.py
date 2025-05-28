#!/usr/bin/env python3
"""
Neural-Decoder Real-time Detection Script

Performs real-time EEG alpha wave detection using trained neural network models.
Supports both traditional threshold-based and advanced neural network approaches.

Usage:
    python scripts/realtime_detection.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import welch
import pyqtgraph as pg
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import winsound

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.realtime_predictor import RealTimePredictor
from utils.config_loader import config


def beep():
    """Simple placeholder for beep sound."""
    print("(Beep!)")
    try:
        winsound.Beep(900, 750)
    except:
        pass  # On non-Windows systems


class EEGDataCollector:
    def __init__(self, mac_address=None):
        """Initialize the EEG data collector with board setup"""
        # Load configuration
        device_config = config.get_device_config()
        if mac_address is None:
            mac_address = device_config.get('mac_address', 'f0:17:3b:41:ec:7d')
            
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
        
        # Get configuration
        sig_config = config.get_signal_processing_config()
        window_size = config.get('ml.advanced.input_size', 600)
        update_size = window_size // 2  # 50% overlap
        update_interval = sig_config.get('update_interval_sec', 1.5)
        
        # Initialize the prediction window with first data points
        initial_data = self.board.get_current_board_data(window_size)[self.channel]
        if len(initial_data) < window_size:
            print("Waiting for enough initial data...")
            time.sleep(2)
            initial_data = self.board.get_current_board_data(window_size)[self.channel]
        
        self.prediction_window = initial_data
        
        try:
            while True:
                # Get most recent points
                new_data = self.board.get_current_board_data(update_size)[self.channel]
                
                if len(new_data) >= update_size:
                    # Move second half of window to first half
                    self.prediction_window[:update_size] = self.prediction_window[update_size:]
                    # Put new data in second half
                    self.prediction_window[update_size:] = new_data
                    
                    # Make prediction on full window
                    prediction, confidence = predictor.predict_point(self.prediction_window)
                    state = "CLOSED" if prediction == 1 else "OPEN"
                    print(f"Eyes: {state}, Confidence: {confidence:.2f}")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nStopping continuous prediction.")
    
    def collect_training_data(self, segment_duration=None):
        """Collect training data with alternating eyes open/closed segments"""
        # Get configuration
        data_config = config.get('data_collection', {})
        if segment_duration is None:
            segment_duration = data_config.get('segment_duration', 15)
        labels_sequence = data_config.get('labels', ['open', 'closed', 'open', 'closed'])
        
        data_segments = []
        labels = []

        try:
            for i, label in enumerate(labels_sequence, 1):
                input(f"Press ENTER to begin segment {i}/{len(labels_sequence)} (Eyes {label.upper()}).")
                segment_data, seg_label = self.collect_segment(segment_duration, label)
                data_segments.append(segment_data)
                labels.append(seg_label)
                beep()

        except KeyboardInterrupt:
            print("Data collection interrupted by user.")

        # Combine all data
        combined_data = np.concatenate(data_segments)
        sample_labels = []
        for label, segment_data in zip(labels, data_segments):
            # Assign 1 for closed eyes, 0 for open eyes
            is_closed = 1 if label == "closed" else 0
            segment_labels = [is_closed] * len(segment_data)
            sample_labels.extend(segment_labels)
        
        # Create DataFrame and save to data directory
        df = pd.DataFrame({'EXG Channel 0': combined_data})
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join('data', 'raw', f'eeg_data_{timestamp}.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=True)
        
        labels_array = np.array(sample_labels, dtype=int)
        
        print(f"\nData collection completed. Saved to: {output_path}")
        return df['EXG Channel 0'].values, labels_array
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'board'):
                if self.board.is_prepared():
                    # Note: Commented out to allow reuse in interactive sessions
                    # self.board.stop_stream()
                    # self.board.release_session()
                    pass
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main real-time detection function."""
    print("Neural-Decoder Real-time Detection")
    print("=" * 40)
    
    # Initialize data collector
    collector = EEGDataCollector()
    
    # Check if trained model exists
    model_path = os.path.join('models', 'real_time_model.pth')
    
    if os.path.exists(model_path):
        print(f"Loading existing model from: {model_path}")
        predictor = RealTimePredictor()
        # Load model if implementation supports it
        # predictor.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        
        # Collect training data
        channel_data, labels = collector.collect_training_data()
        
        # Create and train the predictor
        predictor = RealTimePredictor()
        print("\nTraining neural network...")
        
        # Get training configuration
        ml_config = config.get_ml_config()
        training_config = ml_config.get('training', {})
        
        predictor.train(
            channel_data, 
            labels, 
            epochs=training_config.get('epochs', 20),
            batch_size=training_config.get('batch_size', 32)
        )
        
        # Save the trained model
        os.makedirs('models', exist_ok=True)
        predictor.export_model()
        print(f"Model saved to: {model_path}")
    
    print("\nStarting real-time alpha wave detection...")
    print("Press Ctrl+C to stop.")
    
    # Start continuous prediction
    collector.run_continuous_prediction(predictor)


if __name__ == "__main__":
    main()
