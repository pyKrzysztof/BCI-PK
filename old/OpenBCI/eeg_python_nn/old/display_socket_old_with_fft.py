import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5 import QtWidgets
import numpy as np
import queue
import threading
import argparse
from numpysocket import NumpySocket
import collections



# Data-gathering thread function
def gather_data(port: int, queue: queue.Queue, stop_event: threading.Event):
    packet_count = 0
    socket = NumpySocket()
    try:
        # Connect to the server socket
        socket.connect(("localhost", port))
        while not stop_event.is_set():
            # Receive packets from the server
            packets = socket.recv()
            if len(packets) == 0:
                continue
            
            # Add data to the queue for plotting\
            queue.put(packets)
            
            packet_count = packet_count + 1
            # print(f"N: {packet_count}, Received packets of shape: {packets.shape}")

    except Exception as e:
        print("Error in data gathering:", e)
    finally:
        socket.close()

PACKET_SIZE = 25
SAMPLING_RATE = 250
FREQUENCY_RESOLUTION = 3.33

DATA_SIZE = int(SAMPLING_RATE // FREQUENCY_RESOLUTION)
if DATA_SIZE % PACKET_SIZE != 0:
    DATA_SIZE = int(DATA_SIZE + PACKET_SIZE - (DATA_SIZE % PACKET_SIZE))
FREQUENCY_RESOLUTION = SAMPLING_RATE / DATA_SIZE
print(f"Data size for FFT is: {DATA_SIZE}, achieved resolution of {FREQUENCY_RESOLUTION} Hz and a delay of {1000*DATA_SIZE/SAMPLING_RATE} ms.")

FFT_FREQ_AXIS = np.fft.fftfreq(DATA_SIZE, d=1/SAMPLING_RATE)

# Function to update the plot
def update_plot(data_queue: queue.Queue, curves, time_data, channel_data, marker_data=None, fft_bar_chart=None, fft_data_storage=None):
    try:
        # Get data from the queue
        data = data_queue.get(timeout=0.2)

        # Append the time and marker data (gathering socket must be run with '-markers' parameter)
        time_data.extend(data[-1])
        if marker_data:
            marker_data.extend(data[-2])

        # Update the curves with new data
        for i in range(8):  # 8 channels
            channel_data[i].extend(data[i])
            curve_data = channel_data[i]
            curves[i].setData(x=time_data, y=curve_data)  # Set the data for the curve
        
        # End loop if not enough data to calculate FFT (might be better to calculate FFT with padding when not enough data instead of this if-statement)
        if len(channel_data[0]) < DATA_SIZE:
            return
        
        # Calculate FFT data when there's enough data
        if len(channel_data[0]) >= DATA_SIZE:
            fft_array = []
            for i in range(8):
                windowed_signal = np.array(channel_data[i])[-DATA_SIZE:] * np.hanning(DATA_SIZE)
                fft_array.append(np.abs(np.fft.fft(windowed_signal)))
            
            # Store FFT data is storage is provided.
            if fft_data_storage is not None:
                fft_data_storage.append(fft_array)
            
            # Plot FFT data on the bar chart
            # fft_bar_chart.clear()  # Clear existing data
            colors = [pg.mkBrush(color) for color in ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']]
            for i in range(8):
                fft_bar_chart.addItem(pg.BarGraphItem(x=FFT_FREQ_AXIS[:DATA_SIZE//2], height=fft_array[i][:DATA_SIZE//2], width=0.1, brush=colors[i], name=f'Channel {i+1}'))
        
        
        
    
    # If no data, just continue
    except queue.Empty:
        pass  



def main():
    # Argument parser
    parser = argparse.ArgumentParser("plotting_socket")
    parser.add_argument("--port", type=int, default=9999, required=False)
    args = parser.parse_args()
    
    
    # Setup PyQtGraph
    app = QtWidgets.QApplication([])  # Initialize a Qt application
    win = pg.GraphicsLayoutWidget(show=True)  # Create a graphics layout widget
    win.setWindowTitle("EEG Data")

    # Create plots for 8 channels
    plots = [win.addPlot(row=i, col=0) for i in range(8)]
    curves = [plot.plot() for plot in plots]  # Create curves to update
    
    # Create a bar chart for the FFT data
    fft_plot = win.addPlot(row=8, col=0)
    fft_bar_chart = pg.GraphItem()
    fft_plot.addItem(fft_bar_chart)
    fft_plot.setLabel("bottom", "Frequency", units="Hz")

    # Deques to store data for plotting
    channel_data = [collections.deque(maxlen=750) for _ in range(8)]
    time_data = collections.deque(maxlen=750)
    marker_data = collections.deque(maxlen=750)
    fft_data_storage = []

    # Queue to share data between threads
    data_queue = queue.Queue()

    # Signal to stop the data-gathering thread
    stop_event = threading.Event()
    

    # Set up the data-gathering thread
    gather_thread = threading.Thread(target=gather_data, args=(args.port, data_queue, stop_event))
    gather_thread.start()
    
    _update_plot = lambda: update_plot(data_queue, curves, time_data, channel_data, marker_data=None, fft_bar_chart=fft_bar_chart, fft_data_storage=None)

    # Set up a timer to refresh the plot at a regular interval
    timer = QtCore.QTimer()
    timer.timeout.connect(_update_plot)  # Connect the timer to the update function
    timer.start(100)  # Update every 100 ms

    # Stop data gathering thread when window is closed
    win.closeEvent = lambda _: stop_event.set()

    # Start the PyQt application
    QtWidgets.QApplication.instance().exec_()

    # Ensure the data-gathering thread completes before ending
    gather_thread.join()
    
if __name__ == "__main__":
    main()