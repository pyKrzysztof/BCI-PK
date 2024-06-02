Naming explanation: "training_x_y_z"
x - packet_size used
y - fft_size used
z - detrend type, can be linear or constant.

To get this data I ran:
- data_processor.py on raw (and pipelined) session data
- used prepare_chunks.py and process_chunks.py on resulting files.
- moved the data here

