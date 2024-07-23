import pyxdf
import csv
from collections import deque

def load_xdf(filename, print_header=True):
    data, header = pyxdf.load_xdf(filename)
    if print_header:
        print(header)
    return data

def stream_zip_by_index(data, index):
    stream = data[index]
    time_series = stream["time_series"]
    for timestamp, values in zip(stream["time_stamps"], time_series):
        yield timestamp, values

def almost_equal(a, b, tolerance):
    return abs(a - b) <= tolerance



def base_marker_func_LRS(label):
    return 1 if label == "L" else 2 if label == "R" else 3 if label == "S" else 0

def xdf_to_csv(filename, data_index, marker_index, marker_dict, timestamp_tolerance=0.01, output_file=None):
    """
    'data_index' and 'marker_index' are the stream indices of the <'filename'>.xdf file.\n
    'marker_dict' with values of the marker stream as keys and equivalent numerical values as corresponding values, eg. {'L': 1, 'R': 2}
    """
    data = load_xdf(filename, print_header=False)
    marker_pairs = deque([(ts, marker_dict[marker[0]]) for ts, marker in stream_zip_by_index(data, marker_index) if marker[0] in marker_dict.keys()])

    action_duration = 4
    end_timestamp = 0
    end_marker = 0
    count_marker = 0
    count_end_marker = 0

    if output_file is not None:
        new_filename = output_file
    else:
        new_filename = filename.replace("xdf", "csv")

    with open(new_filename, "w", newline="\n") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for timestamp, values in stream_zip_by_index(data, data_index):
            marker_value = 0
            try:
                marker_timestamp = marker_pairs[0][0]

                if almost_equal(marker_timestamp, timestamp, timestamp_tolerance):
                    marker_value = marker_pairs.popleft()[1]
                    end_timestamp = marker_timestamp + action_duration
                    end_marker = -marker_value
                    # print(timestamp, marker_value)
                    count_marker += 1
            except:
                pass

            if almost_equal(timestamp, end_timestamp, timestamp_tolerance):
                marker_value = end_marker
                end_timestamp = 0
                # print(timestamp, marker_value)
                count_end_marker += 1

            row = [*values, timestamp, marker_value]
            writer.writerow(row)
    try:
        assert count_marker == count_end_marker != 0
        print("Found", count_marker, "marked regions.")
        return 1
    except:
        print("Try increasing the timestamp tolerance, or make sure the stream has all data chunks enclosed.")
        return 0