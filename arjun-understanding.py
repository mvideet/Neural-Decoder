import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import time
import numpy as np
from scipy.signal import welch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes


def compute_alpha_power(data, fs=200):
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)  # Power Spectral Density (PSD)
    alpha_idx = np.where((f >= 8) & (f <= 12))  # Select 8-12 Hz range
    alpha_power = np.sum(Pxx[alpha_idx])  # Sum power in alpha range
    return alpha_power


def main():
    #BoardShim.enable_dev_board_logger()
    
    params = BrainFlowInputParams()
    params.mac_address = "f0:17:3b:41:ec:7d"
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    print("Streaming... press Ctrl+C to stop.")


    exg_channels = BoardShim.get_exg_channels(board_id)
    print("EXG channels:", exg_channels)
    channel_of_interest = exg_channels[0]

    fs = BoardShim.get_sampling_rate(board_id)
    print("Sampling rate:", fs, "Hz")

    window_sec = 3  # seconds
    window_size = int(window_sec * fs)
    overlap = 0.5  # overlap percentage
    time.sleep(window_sec)  # wait for data to accumulate

    
    #     # Create the Qt Application
    # app = QtWidgets.QApplication([])

    # # Create a GraphicsLayoutWidget (instead of GraphicsWindow)
    # win = pg.GraphicsLayoutWidget()
    # win.setWindowTitle('BrainFlow Plot')
    # win.resize(800, 600)
    # plot_item = win.addPlot(title="BrainFlow Data")  
    # plot_item.plot([1, 2, 3, 4, 5], [10, 20, 15, 30, 25])  # X and Y data

    # win.show()
 
    try:
        while True:
            data = board.get_current_board_data(window_size)
            # print("Data:", data[channel_of_interest], "Shape:", data.shape[channel_of_interest])

            time.sleep(window_sec * overlap)
    except:
        pass



if __name__ == "__main__":
    main()