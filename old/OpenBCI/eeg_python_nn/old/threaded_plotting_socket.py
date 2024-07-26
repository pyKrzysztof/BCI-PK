import time
import queue
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import collections
import argparse
from numpysocket import NumpySocket

# Argument parser for command-line options
parser = argparse.ArgumentParser("plotting_socket")
parser.add_argument("--port", type=int, default=9999, required=False)
args = parser.parse_args()

# Deque to store data for plotting
channels = collections.deque(maxlen=250)
times = collections.deque(maxlen=250)

# Queue to share data between threads
data_queue = queue.Queue()

# Data-gathering thread function
def gather_data(socket: NumpySocket, queue: queue.Queue):
    packet_count = 0
    try:
        # Connect to the server socket
        socket.connect(("localhost", args.port))
        while True:
            # Receive packets from the server
            packets = socket.recv()
            if len(packets) == 0:
                continue
            
            # Add data to the queue for plotting
            queue.put(packets)
            packet_count = packet_count + 1
            print(f"N: {packet_count}, Received packets of shape: {packets.shape}")
            
    except Exception as e:
        print("Error in data gathering:", e)
    finally:
        socket.close()

# Function to plot data from the queue
def plot_data(fig, axes):
    try:
        while True:
            # Retrieve packets from the queue
            packets = data_queue.get()  # Blocking call until new data arrives
            
            # Update plot data
            for packet in packets:
                channels.append(packet[:-1])
                times.append(packet[-1])

            # Update plots
            for i, ax in enumerate(axes):
                ax.clear()
                ax.plot(times, [data[i] for data in channels])
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            fig.canvas.draw()
            plt.pause(0.1)  # Pause to refresh the plot
    except Exception as e:
        print("Error in plot update:", e)

# Set up the initial plot
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(4, 8), constrained_layout=True)
fig.canvas.draw()

# Start the data-gathering thread
with NumpySocket() as socket:
    # Start a thread to gather data
    gather_thread = threading.Thread(target=gather_data, args=(socket, data_queue))
    gather_thread.start()

    # Run the plot_data function in the main thread
    plot_data(fig, axes)

    # Ensure the data-gathering thread completes before ending
    gather_thread.join()

plt.show()  # Display the plot at the end
