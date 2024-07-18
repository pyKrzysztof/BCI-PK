import pyxdf
import csv
from collections import deque

filename = "12-07-2024_LRS_1H_30T_Nr2.xdf"
data, header = pyxdf.load_xdf(filename)


def stream_zip(index):
    stream = data[index]
    time_series = stream['time_series']
    for timestamp, values in zip(stream['time_stamps'], time_series):
        yield timestamp, values

new_filename = "converted.csv"

# 0 - unfiltered
# 1 - filtered
# 2 - fft
# 3 - markers

# for ts, v in stream_zip(3):
    # print(v, ts)

almost_equal = lambda a, b, tolerance: abs(a - b) <= tolerance
label_func = lambda label: 1 if label == "L" else 2 if label == "R" else 3 if label == "S" else 0
marker_pairs = deque([(ts, label_func(marker[0])) for ts, marker in stream_zip(3) if marker[0] in 'LRS'])
# print(marker_timestamps)
# print(marker_pairs[0][0])
action_duration = 4
end_timestamp = 0
end_marker = 0

count_marker = 0
count_end_marker = 0

with open(new_filename, 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    for timestamp, values in stream_zip(0):
        marker_value = 0
        try:
            marker_timestamp = marker_pairs[0][0]

            if almost_equal(marker_timestamp, timestamp, 0.003):
                marker_value = marker_pairs.popleft()[1]
                end_timestamp = marker_timestamp + action_duration
                end_marker = -marker_value
                print(timestamp, marker_value)
                count_marker += 1
        except:
            pass

        if almost_equal(timestamp, end_timestamp, 0.1):
            marker_value = end_marker
            end_timestamp = 0
            print(timestamp, marker_value)
            count_end_marker += 1

        row = [*values, timestamp, marker_value]
        writer.writerow(row)

assert count_marker == count_end_marker