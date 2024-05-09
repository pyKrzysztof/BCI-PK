import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5 import QtWidgets
import numpy as np
import queue
import threading
import argparse
from numpysocket import NumpySocket
import collections
import time





# Data-gathering thread function
def gather_data(port: int, data_queue: queue.Queue, stop_event: threading.Event):
    
    # Connect to the server socket
    socket = NumpySocket()
    
    threshold = 2
    connected = False

    # Main data loop, will attempt to reconnect if connection is lost. 
    # (Currently getting BAD_FILE_DESCRIPTOR errors when reconnecting to a port that was already 
    # open once during execution of this program, manually closing the socket doesn't work. 
    # It might be due to how the other side (server) handles the socket. )
    last_packet_time = time.time()
    while not stop_event.is_set():
        # Attempt connection if not connected
        if not connected:
            try:
                socket.connect(("localhost", port))
                print("Connected to server.")
                connected = True
            except Exception as e:
                print(e)
                print("No server connection, retrying in 3 seconds..")
                time.sleep(3)

        if connected:
            try:
                # Receive packets from the server
                packets = socket.recv()
                if len(packets) == 0:
                    # Close the socket if no new packets in 'threshold' seconds.
                    if time.time() - last_packet_time > threshold:
                        socket.close()
                        connected = False
                    continue
                # Measure time
                last_packet_time = time.time()
                # Add data to the queue for plotting
                data_queue.put(packets)

            except Exception as e:
                socket.close()
                connected = False


    # Close the socket
    socket.close()





# Function to update the plots
def update_plot(data_queue: queue.Queue, curves, fft_curves, time_data, channel_data, samples_to_display, fft_data_size, fft_freq_axis):
    try:
        # Get data from the queue
        data = data_queue.get(timeout=0.2)

        # Append the time data
        time_data.extend(data[-1])

        # Update the time-domain plots
        for i in range(8):  # 8 channels
            channel_data[i].extend(data[i])
            curves[i].setData(x=time_data, y=list(channel_data[i])[-samples_to_display:])

        # If there's enough data, calculate and update the FFT plots
        if len(channel_data[0]) >= fft_data_size:
            for i in range(8):
                windowed_signal = np.array(channel_data[i])[-fft_data_size:] * np.hanning(fft_data_size)
                fft_result = np.abs(np.fft.fft(windowed_signal))[:fft_data_size//2]
                fft_curves[i].setData(x=fft_freq_axis, y=fft_result)

    except queue.Empty:
        pass




def main():
    # Argument parser
    parser = argparse.ArgumentParser("plotting_socket")
    parser.add_argument("--port", type=int, default=9999, required=False)
    parser.add_argument("--packet_size", type=int, default=25, required=False)
    parser.add_argument("--fft_res", type=float, default=3.33, help="Target frequency resolution for fft, will be rounded to the nearest possible value for the given packet size.", required=False)
    parser.add_argument("--fs", type=int, default=250, help="Board sampling rate.", required=False)
    parser.add_argument("--time_frame", type=float, default=3.0, help="Amount of time displayed at any given moment on the graph.", required=False)
    args = parser.parse_args()

    # Setup PyQtGraph
    app = QtWidgets.QApplication([])  # Initialize a Qt application
    win = pg.GraphicsLayoutWidget(show=True)  # Create a graphics layout widget
    win.setWindowTitle("EEG Data")
    
    # Define graph channel colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']

    # Create plots for 8 channels
    plots = [win.addPlot(row=i, col=0) for i in range(8)]
    curves = [plot.plot(pen=pg.mkPen(color)) for plot, color in zip(plots, colors)]  # Create curves to update

    # Create a single plot for FFT data
    fft_plot = win.addPlot(row=8, col=0)  # Common plot for FFT data
    fft_plot.setLabel("bottom", "Frequency", units="Hz")

    fft_data_size = int(args.fs // args.fft_res)
    if fft_data_size % args.packet_size != 0:
        fft_data_size = int(fft_data_size + args.packet_size - (fft_data_size % args.packet_size))

    fft_res = args.fs / fft_data_size
    fft_freq_axis = np.fft.fftfreq(fft_data_size, d=1/args.fs)[:fft_data_size//2]
    print(f"Achieved frequency resolution of {fft_res} Hz and a delay of {1000*fft_data_size/args.fs} ms.")

    # Initialize a list of PlotDataItems for FFT curves
    fft_curves = [fft_plot.plot(pen=pg.mkPen(color)) for color in colors]

    # Set relative height of rows, giving more space to the last row
    win.ci.layout.setRowStretchFactor(8, 3)  # Stretch factor for the FFT plot
    for i in range(8):
        win.ci.layout.setRowStretchFactor(i, 1)  # Lower stretch factor for other plots

    # Calculate the stored data size for displaying on the plot
    samples_to_display = int(args.fs * args.time_frame)

    # Deques to store data for plotting
    channel_data = [collections.deque(maxlen=max(samples_to_display, fft_data_size)) for _ in range(8)]
    time_data = collections.deque(maxlen=samples_to_display)

    # Queue to share data between threads
    data_queue = queue.Queue()

    # Signal to stop the data-gathering thread
    stop_event = threading.Event()

    # Set up the data-gathering thread
    gather_thread = threading.Thread(target=gather_data, args=(args.port, data_queue, stop_event))
    gather_thread.start()

    # Define the lambda function to update plots
    _update_plot = lambda: update_plot(data_queue, curves, fft_curves, time_data, channel_data, samples_to_display, fft_data_size, fft_freq_axis)

    # Set up a timer to refresh the plot at a regular interval
    timer = QtCore.QTimer()
    timer.timeout.connect(_update_plot)  # Connect the timer to the update function
    timer.start(100)  # Update every 100 ms

    # Stop data-gathering thread when window is closed
    win.closeEvent = lambda _: stop_event.set()

    # Start the PyQt application
    QtWidgets.QApplication.instance().exec_()

    # Ensure the data-gathering thread completes before ending
    gather_thread.join()

if __name__ == "__main__":
    main()
