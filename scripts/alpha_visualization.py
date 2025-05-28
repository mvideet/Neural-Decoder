#!/usr/bin/env python3
"""
Neural-Decoder Alpha Visualization Script

Real-time visualization of EEG signals and alpha wave detection.
Displays live EEG waveforms, alpha power, and detection status.

Usage:
    python scripts/alpha_visualization.py
"""

import sys
import os
import time
import numpy as np
from scipy.signal import welch
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
#from PyQt5.QTwidgets import QLabel, QVBoxLayout, QWidget #is this needed?
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_loader import config


def compute_alpha_power(data, fs):
    """Compute alpha power (8-12 Hz) using Welch's method."""
    alpha_config = config.get_signal_processing_config().get('alpha_band', {})
    low_freq = alpha_config.get('low_freq', 8.0)
    high_freq = alpha_config.get('high_freq', 12.0)
    
    f, Pxx = welch(data, fs=fs, nperseg=fs // 2)
    alpha_idx = np.where((f >= low_freq) & (f <= high_freq))
    alpha_power = np.sum(Pxx[alpha_idx])
    return alpha_power


class AlphaVisualizationApp:
    def __init__(self, board_shim):
        """Initialize the visualization application."""
        # Configuration
        self.config = config
        vis_config = self.config.get_visualization_config()
        sig_config = self.config.get_signal_processing_config()
        
        # Alpha level threshold
        self.alpha_threshold = vis_config.get('alpha_threshold', 0.003726)

        # Board setup
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.channel_of_interest = BoardShim.get_exg_channels(self.board_id)[0]
        
        # Window parameters
        self.window_size = sig_config.get('window_size_sec', 3.0)
        self.num_points = int(self.window_size * self.sampling_rate)
        self.overlap = sig_config.get('overlap_percent', 0.5)
        self.update_interval = sig_config.get('update_interval_sec', 1.5)
        self.alpha_window = vis_config.get('alpha_window_sec', 60) // self.update_interval

        # Create the PyQt Application
        self.app = QtWidgets.QApplication([])

        # Main widget setup
        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.main_widget.setWindowTitle(vis_config.get('window_title', 'Neural-Decoder Alpha Visualization'))
        self.main_widget.resize(
            vis_config.get('plot_width', 800), 
            vis_config.get('plot_height', 600)
        )

        # PyQtGraph widget
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Real-time EEG and Alpha Power Analysis')
        self.layout.addWidget(self.win)

        # EEG Time Series plot
        self.eeg_plot = self.win.addPlot(row=0, col=0)
        self.eeg_plot.setLabel('left', f'Channel {self.channel_of_interest}')
        self.eeg_plot.setLabel('bottom', 'Sample Points')
        self.eeg_plot.setTitle('EEG Time Series')
        self.eeg_curve = self.eeg_plot.plot()

        # Alpha Power plot
        self.alpha_plot = self.win.addPlot(row=1, col=0)
        self.alpha_plot.setLabel('left', 'Alpha Power')
        self.alpha_plot.setLabel('bottom', 'Time Windows')
        self.alpha_plot.setTitle('Alpha Power Over Time')
        self.alpha_curve = self.alpha_plot.plot()

        # Status label for alpha detection
        self.status_label = QtWidgets.QLabel("Alpha waves NOT detected (initializing...)")
        self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.status_label)

        # Threshold line on alpha plot
        self.threshold_line = pg.InfiniteLine(
            angle=0, movable=False, 
            pen=pg.mkPen('r', style=QtCore.Qt.DashLine)
        )
        self.threshold_line.setPos(self.alpha_threshold)
        self.alpha_plot.addItem(self.threshold_line)

        # Show the main widget
        self.main_widget.show()

    def run_loop(self):
        """Continuous loop that fetches, processes, and plots data."""
        alpha_power_list = []
        
        while True:
            try:
                # Fetch data
                data = self.board_shim.get_current_board_data(self.num_points)[self.channel_of_interest]
                
                if data.shape[0] >= self.num_points:
                    # Process the data
                    processed_data = self._process_eeg_data(data)
                    
                    # Compute alpha power
                    alpha_val = compute_alpha_power(processed_data, fs=self.sampling_rate)
                    alpha_power_list.append(alpha_val)
                    
                    # Maintain window size
                    if len(alpha_power_list) > self.alpha_window:
                        alpha_power_list.pop(0)

                    # Update plots
                    self._update_plots(processed_data, alpha_power_list, alpha_val)

                # Process GUI events
                QtWidgets.QApplication.processEvents()

                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in visualization loop: {e}")
                break

    def _process_eeg_data(self, data):
        """Apply signal processing to raw EEG data."""
        # Get signal processing configuration
        sig_config = self.config.get_signal_processing_config()
        bandpass_config = sig_config.get('bandpass', {})
        notch_config = sig_config.get('notch', {})
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Normalize data
        processed_data = 2 * (processed_data - np.min(processed_data)) / (np.max(processed_data) - np.min(processed_data)) - 1

        # Apply filters
        DataFilter.detrend(processed_data, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(
            processed_data, self.sampling_rate,
            bandpass_config.get('low_freq', 3.0),
            bandpass_config.get('high_freq', 45.0),
            bandpass_config.get('order', 2),
            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
        )
        DataFilter.perform_bandstop(
            processed_data, self.sampling_rate,
            notch_config.get('low_freq', 58.0),
            notch_config.get('high_freq', 62.0),
            notch_config.get('order', 2),
            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
        )
        
        return processed_data

    def _update_plots(self, eeg_data, alpha_power_list, current_alpha):
        """Update all plots with new data."""
        # Update EEG time series plot
        self.eeg_curve.setData(eeg_data)

        # Update alpha power plot
        self.alpha_curve.setData(alpha_power_list)
        
        # Print alpha power value
        print(f"Channel {self.channel_of_interest} Alpha Power: {current_alpha:.2e}")

        # Update status label based on threshold comparison
        if current_alpha > self.alpha_threshold:
            self.status_label.setText("✅ Alpha waves DETECTED (Eyes likely closed)")
            self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            self.status_label.setText("❌ Alpha waves NOT detected (Eyes likely open)")
            self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")


def main():
    """Main function to initialize and run the visualization."""
    print("Neural-Decoder Alpha Visualization")
    print("=" * 40)
    
    # Load device configuration
    device_config = config.get_device_config()
    
    # Initialize BrainFlow
    params = BrainFlowInputParams()
    params.mac_address = device_config.get('mac_address', 'f0:17:3b:41:ec:7d')
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    board_shim = BoardShim(board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream()
        print("Session prepared. Starting real-time visualization...")

        # Create and run visualization
        viz_app = AlphaVisualizationApp(board_shim)
        viz_app.run_loop()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        try:
            board_shim.stop_stream()
            if board_shim.is_prepared():
                board_shim.release_session()
            print("BrainFlow session cleaned up.")
        except:
            pass


if __name__ == "__main__":
    main()
