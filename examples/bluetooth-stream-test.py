import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def main():
    # (Optional) Enable BrainFlow’s logger for debugging
    BoardShim.enable_dev_board_logger()

    # 1) Prepare parameters
    params = BrainFlowInputParams()
    # If you know your Ganglion's BLE MAC, specify it:
    # On Windows, it might be something like '00:07:80:AB:CD:EF'
    # On Mac, might be 'F1:E2:D3:C4:B5:A6'
    params.mac_address = "f0:17:3b:41:ec:7d"

    # 2) Create a 'native' Ganglion board object
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    board = BoardShim(board_id, params)

    # 3) Initialize BLE session
    board.prepare_session()

    # 4) Start streaming
    board.start_stream()  # optionally pass in buffer size or streamer_params
    print("Streaming for 10 seconds...")

    # 5) Wait while data accumulates
    time.sleep(10)

    # 6) Retrieve all board data
    data = board.get_board_data()

    # 7) Stop streaming and close
    board.stop_stream()
    board.release_session()

    # 'data' is a NumPy 2D array: rows = channels, cols = samples
    print("Data shape:", data.shape)
    print(data)

if __name__ == "__main__":
    main()
