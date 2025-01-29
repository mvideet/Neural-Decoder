import time
import numpy as np
from scipy.signal import welch
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
#from PyQt5.QTwidgets import QLabel, QVBoxLayout, QWidget #is this needed?
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


def compute_alpha_power(data, fs):
    """Compute alpha power (8-12 Hz) using Welch's method."""
    f, Pxx = welch(data, fs=fs, nperseg=fs // 2)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    alpha_power = np.sum(Pxx[alpha_idx])
    return alpha_power


class Graph:
    def __init__(self, board_shim):

        # alpha level threshold
        self.alpha_threshold = 3.25e-5

        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.channel_of_interest = BoardShim.get_exg_channels(self.board_id)[0]
        # Window size (seconds) and sample count
        self.window_size = 3
        self.num_points = int(self.window_size * self.sampling_rate)
        # Update interval (seconds) -> 1.5s for 50% overlap
        self.overalp = .5
        self.update_interval = self.window_size * self.overalp

        self.alpha_window = 60 // self.update_interval # 60s window


                # Create the PyQt Application
        self.app = QtWidgets.QApplication([])

        # 1) Main widget to hold layout
        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.main_widget.setWindowTitle("BrainFlow Plot with Alpha Status")
        self.main_widget.resize(800, 600)

        # 2) PyQtGraph widget
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('BrainFlow Plot (3s Window, Updates Every 1.5s)')
        self.layout.addWidget(self.win)

        # 3) EEG Time Series plot
        self.p = self.win.addPlot(row=0, col=0)
        self.p.setLabel('left', f'Channel {self.channel_of_interest}')
        self.p.setTitle('EEG Time Series')
        self.curve = self.p.plot()

        # 4) Alpha Power plot
        self.alpha = self.win.addPlot(row=1, col=0)
        self.alpha.setLabel('left', 'Alpha Power')
        self.alpha.setTitle('Alpha Power')
        self.alpha_curve = self.alpha.plot()

        # 5) Status label for alpha detection
        self.status_label = QtWidgets.QLabel("Alpha waves NOT detected (initial)")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.layout.addWidget(self.status_label)

        # 6) threshold line added to alpha plot:
        self.threshold_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.threshold_line.setPos(self.alpha_threshold)
        # Add it to the alpha plot
        self.alpha.addItem(self.threshold_line)

        # Show the main widget
        self.main_widget.show()




    def run_loop(self):
        """Continuous loop that fetches, processes, and plots data."""
        alpha_power_list = []
        while True:
            # 1) Fetch last 3s of data
            data = self.board_shim.get_current_board_data(self.num_points)[self.channel_of_interest]
            if data.shape[0] >= self.num_points:
                # 2) Basic filtering
                DataFilter.detrend(data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(
                    data, self.sampling_rate, 3.0, 45.0, 2,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                )
                DataFilter.perform_bandstop(
                    data, self.sampling_rate, 58.0, 62.0, 2,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                )

                # 3) Compute alpha power
                alpha_val = compute_alpha_power(data, fs=self.sampling_rate)
                alpha_power_list.append(alpha_val)
                if len(alpha_power_list) > self.alpha_window:
                    alpha_power_list.pop(0)

                # 4) Update alpha power plot
                self.alpha_curve.setData(alpha_power_list)
                print(f"Channel {self.channel_of_interest} Alpha Power: {alpha_val:.2e}")

                # 5) Update EEG Time Series plot
                self.curve.setData(data)

                # 6) Compare alpha_val to threshold & update label
                if alpha_val > self.alpha_threshold:
                    self.status_label.setText("Alpha waves DETECTED")
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                else:
                    self.status_label.setText("Alpha waves NOT detected")
                    self.status_label.setStyleSheet("color: red; font-weight: bold;")

            # Process GUI events so the GUI stays responsive
            QtWidgets.QApplication.processEvents()

            # Sleep for 1.5s -> 50% overlap on a 3s window
            time.sleep(self.update_interval)

def main():
    # Initialize BrainFlow with minimal params (adjust MAC as needed)
    params = BrainFlowInputParams()
    params.mac_address = "f0:17:3b:41:ec:7d"
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    board_shim = BoardShim(board_id, params)

    board_shim.prepare_session()
    board_shim.start_stream()
    print("Session prepared. Starting real-time plot loop...")

    graph = Graph(board_shim)
    try:
        graph.run_loop()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        board_shim.stop_stream()
        if board_shim.is_prepared():
            board_shim.release_session()


if __name__ == "__main__":
    main()
