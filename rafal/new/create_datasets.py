
import mybci
from mybci.data_processor import Config, DataProcessor
from mybci.feature_extraction import Features, Filters
from mybci.feature_extraction import get_data_filter, get_feature_extractor
from mybci.new_processor import NewProcessor, NewConfig

config = Config()

config.name = 'Krzychu'
config.session_folders = ['session_me/']
config.excluded_session_files = ['.xdf', 'Zone']
config.dataset_directory = 'datasets/'

config.action_markers = [1, 2]
config.channel_column_ids = list(range(1, 9 - 2))
config.buffer_size = 128
config.packet_size = 64
config.feature_size = 128
config.chunk_offset = 250

config.filter_function = get_data_filter([
    # Filters.detrend(),
    Filters.bandpass(1, 45, 3),
    Filters.bandstop(49, 51, 4),
    Filters.wavelet_denoise(3)
], value_scale=1e-6)

config.feature_function = get_feature_extractor(
    features=[
        Features.WAVELETS,
    ]
)

# DataProcessor(config).process()
