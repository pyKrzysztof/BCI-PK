import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5 import QtWidgets
import numpy as np
import time
import queue
import threading
import argparse
from numpysocket import NumpySocket
import collections

# Argument parser
parser = argparse.ArgumentParser("plotting_socket")
parser.add_argument("--port", type=int, default=9999, required=False)
args = parser.parse_args()

# Queue to share data between threads
data_queue = queue.Queue()

# Signal to stop the data-gathering thread
stop_event = threading.Event()

# Data-gathering thread function
def gather_data(port, queue: queue.Queue):
    # Deque to store data for plotting
    channels = collections.deque(maxlen=750)
    times = collections.deque(maxlen=750)
    
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
            for packet in packets:
                channels.append(packet[:-1])
                times.append(packet[-1])
            queue.put((channels, times))
            
            packet_count = packet_count + 1
            print(f"N: {packet_count}, Received packets of shape: {packets.shape}")

    except Exception as e:
        print("Error in data gathering:", e)
    finally:
        socket.close()

# Setup PyQtGraph
app = QtWidgets.QApplication([])  # Initialize a Qt application
win = pg.GraphicsLayoutWidget(show=True)  # Create a graphics layout widget
win.setWindowTitle("Real-Time EEG Plot")

# Create plots for 8 channels
plots = [win.addPlot(row=i, col=0) for i in range(8)]
curves = [plot.plot() for plot in plots]  # Create curves to update


# Function to update the plot
def update_plot():
    try:
        # Get data from the queue
        channels, times = data_queue.get(timeout=0.2)
        
        for i in range(8):  # 8 channels
            # Update the curve with new data
            curve_data = [row[i] for row in channels]
            curves[i].setData(x=times, y=curve_data)  # Set the data for the curve
    except queue.Empty:
        pass  # If no data, just continue

# Set up the data-gathering thread
gather_thread = threading.Thread(target=gather_data, args=(args.port, data_queue))
gather_thread.start()

# Set up a timer to refresh the plot at a regular interval
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)  # Connect the timer to the update function
timer.start(100)  # Update every 100 ms

# Stop data gathering thread when window is closed
win.closeEvent = lambda _: stop_event.set()

# Start the PyQt application
QtWidgets.QApplication.instance().exec_()

# Ensure the data-gathering thread completes before ending
gather_thread.join()