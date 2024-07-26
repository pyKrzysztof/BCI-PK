import time
from numpysocket import NumpySocket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import threading
import collections

parser = argparse.ArgumentParser("plotting_socket")
parser.add_argument("--port", type=int, default=9999, required=False)
args = parser.parse_args()


# data_queue = threading.
channels = collections.deque(maxlen=250)
times = collections.deque(maxlen=250)

def plot_data(socket : NumpySocket, epochStart, fig, axes):
    packet_count = 0
    try:
        while True:
            # Receive packets from the server
            packets = socket.recv()
            if len(packets) == 0:
                continue
            packet_count = packet_count + 1
            print(f"N: {packet_count}, Received packets of shape: {packets.shape}")
            for packet in packets:
                channels.append(packet[:-1])
                times.append(packet[-1] - epochStart)

            for i, ax in enumerate(axes):
                ax.clear()
                ax.plot(times, [data[i] for data in channels])
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            fig.canvas.draw()

    except Exception as e:
        raise e
    finally:
        socket.close()

# Wait for the initial graph to appear.
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(4, 8), constrained_layout=True)
fig.canvas.draw()


# Connect to the socket and plot data
with NumpySocket() as socket:
    try:
        sinceEpochStart = time.time()
        socket.connect(("localhost", args.port))
        plot_data(socket, sinceEpochStart, fig, axes)
    except Exception as e:
        raise e

plt.show()