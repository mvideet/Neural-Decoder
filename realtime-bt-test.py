import argparse
import logging

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

from PyQt5 import QtCore, QtWidgets


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)

        # Refresh interval (ms)
        self.update_speed_ms = 50
        # Window size in seconds
        self.window_size = 4
        self.num_points = int(self.window_size * self.sampling_rate)

        # Create the Qt Application
        self.app = QtWidgets.QApplication([])

        # Create a GraphicsLayoutWidget (instead of GraphicsWindow)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('BrainFlow Plot')
        self.win.resize(800, 600)
        self.win.show()

        # Set up timeseries plots
        self._init_timeseries()

        # QTimer for periodic updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        # Block here until the window is closed
        self.app.exec_()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []

        # Create one row per EXG channel
        for i in range(len(self.exg_channels)):
            # Add a new PlotItem in row i
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)

            if i == 0:
                p.setTitle('TimeSeries Plot')

            self.plots.append(p)

            # Create a curve to update with new data
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        """This function is called by the QTimer every self.update_speed_ms ms."""
        # Get the latest samples from BrainFlow
        data = self.board_shim.get_current_board_data(self.num_points)

        # Process each EXG channel
        for idx, channel in enumerate(self.exg_channels):
            # Basic signal filtering
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                data[channel], self.sampling_rate, 3.0, 45.0, 2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )
            DataFilter.perform_bandstop(
                data[channel], self.sampling_rate, 48.0, 52.0, 2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )
            DataFilter.perform_bandstop(
                data[channel], self.sampling_rate, 58.0, 62.0, 2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )

            # Update the curve
            self.curves[idx].setData(data[channel].tolist())

        # Process pending Qt events to keep the GUI responsive
        QtWidgets.QApplication.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, help='BLE discovery timeout', default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol', default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', default='')
    parser.add_argument('--serial-port', type=str, help='serial port', default='')
    parser.add_argument('--mac-address', type=str, help='mac address', default='')
    parser.add_argument('--other-info', type=str, help='other info', default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', default='')
    parser.add_argument('--serial-number', type=str, help='serial number', default='')
    parser.add_argument('--board-id', type=int, help='board id', default=BoardIds.GANGLION_NATIVE_BOARD)
    parser.add_argument('--file', type=str, help='file', default='')
    parser.add_argument('--master-board', type=int, help='master board id', default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board_shim = BoardShim(args.board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params)
        print('Session prepared. Starting real-time plot window...')
        Graph(board_shim)
    except BaseException:
        print('Exception')
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    print('script is running...')
    main()
