from mybci.system import SystemConfig
from mybci.feature_extraction import Filters, Features
from brainflow import BoardIds

config = SystemConfig()
config.path = "/home/chris/workspace/mybci-lab/examples/example_system/"
config.name = "Data Processing"
config.board_type = BoardIds.CYTON_BOARD
config.board_port = "/dev/ttyUSB0"

config.actions = ["Pull", "Push", "Relax"]
config.packet_size = 32
config.feature_size = 128
config.sampling_rate = 250
config.process_overlaps = False

config.channels_accel = [10, 11, 12]
config.channels_eeg = [1, 2, 3, 4, 5, 6, 7, 8]
config.electrode_config = []

config.filters = [(Filters.BANDPASS, 1, 45, 3, 250), (Filters.BANDSTOP, 49, 51, 4, 250), (Filters.WAVELET_DENOISE, 3)]

config.features = [Features.WAVELETS, ]

config.separator = "	"
