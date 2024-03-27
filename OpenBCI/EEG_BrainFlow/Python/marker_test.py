import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port', required=True, default='')
    port = parser.parse_args().serial_port
    
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = port
    
    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    
    board.start_stream()
    
    for i in range(10):
        time.sleep(1)
        board.insert_marker(i + 1)
    time.sleep(2)
    
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    
    print(data)



if __name__ == "__main__":
    main()