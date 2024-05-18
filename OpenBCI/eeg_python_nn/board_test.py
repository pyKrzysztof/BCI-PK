from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets    
import time


params = BrainFlowInputParams()
params.serial_port = "/dev/ttyUSB0"
board = BoardShim(BoardIds.CYTON_BOARD, params)

board.prepare_session()

board.start_stream()

time.sleep(5)

board.stop_stream()
board.release_session()