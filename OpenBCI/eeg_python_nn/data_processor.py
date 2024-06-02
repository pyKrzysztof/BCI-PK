import numpy as np
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, FilterTypes
from brainflow.board_shim import BoardShim, BoardIds


SAMPLING_RATE = 250
ROWS = 24

# does nothing
def raw(data):
    return data

# detrend + bandpass + bandstops
def process_1(data):
    DataFilter.detrend(data, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(data, SAMPLING_RATE, 4.0, 45.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    DataFilter.perform_bandstop(data, SAMPLING_RATE, 48.0, 52.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    DataFilter.perform_bandstop(data, SAMPLING_RATE, 58.0, 62.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    return data


# wrapper function
def process_channel_data(func, channels):

    def new_func(data):
        for i in channels:
            data[i] = func(data[i])
        
        return data
    
    return new_func

# function to process data packets
def process_data(data_packet, full_data_pointer, process_handles):
    n_processes = len(process_handles)

    for i in range(n_processes):
        data_copy = np.copy(data_packet, order="C")
        process_handles[i](data_copy)
        # print(data_copy.shape)
        full_data_pointer[i] = np.array(np.concatenate((full_data_pointer[i], data_copy), axis=1), order="C")

    return full_data_pointer


def process_from_file(filepath, packet_size=32, process_list=[raw, ]):
    file_data = DataFilter.read_file(filepath)

    channels = BoardShim.get_eeg_channels(board_id=BoardIds.CYTON_BOARD)
    handles = [process_channel_data(func, channels) for func in process_list]

    def get_packet(file_data):
        try:
            indices = list(range(0, packet_size, 1))
            data = np.array(file_data[:, indices], order="C")
            file_data = np.delete(file_data, indices, axis=1)
        except IndexError:
            if len(file_data) > 0:
                data = np.array(file_data, order="C")
                file_data = None
        except:
            print("no more data")
            data = None
        finally:
            return file_data, data
    
    full_data = [np.empty((ROWS, 1), order="C") for _ in range(len(handles))]
    file_data, packet = get_packet(file_data)
    current_raw_data = np.copy(packet, order="C")
    packet_count = 1
    while packet is not None:
        current_raw_data = np.array(np.concatenate((current_raw_data, packet), axis=1), order="C")
        full_data = process_data(packet, full_data, handles)
        file_data, packet = get_packet(file_data)
        packet_count = packet_count + 1
        print(packet_count)

    for i, data in enumerate(full_data):
        DataFilter.write_file(data, f"process_{i}_data.csv", "w")



if __name__ == "__main__":
    process_from_file("dane/dane_50-50lewo_prawo.csv", packet_size=64, process_list=[process_1,])
