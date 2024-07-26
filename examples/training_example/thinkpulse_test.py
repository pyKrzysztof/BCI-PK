import mybci.recording
from brainflow import BoardIds


config = ['x1040010X', 'x2040010X', 'x3040010X', 'x4040010X', 'x5040010X', 'x6040010X', "x7040010X", "x8040010X"]

datafile_name = mybci.recording.record_data(
    BoardIds.CYTON_BOARD.value, "/dev/ttyUSB0", 
    32, 
    pipeline="pipelines/pipeline_normal.json", 
    output_name="thinkpulse_test.csv", 
    electrode_config=config, 
    simulated=False
)

