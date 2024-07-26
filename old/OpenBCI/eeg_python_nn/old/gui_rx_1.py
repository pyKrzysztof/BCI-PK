import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import queue
import threading
import argparse
from numpysocket import NumpySocket
import collections
import time
import sys


SAMPLING_FREQUENCY = 250
PACKET_SIZE = 25


# Data-gathering thread function
def socket_connection(address, port, data_queue, stop_event, callback):
    socket = NumpySocket()
    threshold = 2
    last_packet_time = time.time()
    
    try:
        socket.connect((address, port))
        print("Connected to server.")
    except Exception as e:
        print("No server connection.")
        callback()
        return
    
    while not stop_event.is_set():
        try:
            packets = socket.recv()
            if len(packets) == 0:
                if time.time() - last_packet_time > threshold:
                    socket.shutdown(2)
                    break
                continue

            data_queue.put(packets)
            last_packet_time = time.time()
        except:
            socket.shutdown(2)
            break

    print("Closing connection.")
    socket.close()
    callback()

# Function to update the plots
def collect_data(data_queue: queue.Queue, curves, fft_curves, time_data, channel_data, samples_to_display, fft_data_size, fft_freq_axis):
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
                windowed_signal = np.array(channel_data[i])[-fft_data_size:] * np.hanning(fft_data_size)
                fft_result = np.abs(np.fft.fft(windowed_signal))[:fft_data_size//2]
                fft_curves[i].setData(x=fft_freq_axis, y=fft_result)

        
                

    except queue.Empty:
        pass

# Setup global socket_thread and fft parameters
socket_thread = None
fft_freq_axis = None
fft_data_size = None
plot_live = True

def main():
    # Argument parser
    parser = argparse.ArgumentParser("plotting_socket")
    parser.add_argument("--time_frame", type=float, default=3.0, help="Amount of time displayed at any given moment on the graph.", required=False)
    args = parser.parse_args()

    # Setup PyQtGraph
    app = QtWidgets.QApplication([])  # Initialize a Qt application
    win = pg.GraphicsLayoutWidget(show=True)  # Create a graphics layout widget
    win.setWindowTitle("EEG Data")

    # Add widgets for port and address input fields and a connect button
    port_edit = QtWidgets.QLineEdit("9999")
    address_edit = QtWidgets.QLineEdit("localhost")
    fft_res_label = QtWidgets.QLabel("FFT Resolution:")
    fft_res_edit = QtWidgets.QLineEdit("2.5")
    fft_set_button = QtWidgets.QPushButton("Set FFT Resolution")
    connect_button = QtWidgets.QPushButton("Connect")

    # FFT Parameters and calculating function
    def set_new_fft_data_size():
        global fft_freq_axis
        global fft_data_size
        freq_res = float(fft_res_edit.text())
        fft_data_size = int(SAMPLING_FREQUENCY // freq_res)
        if fft_data_size % PACKET_SIZE != 0:
            fft_data_size = int(fft_data_size + PACKET_SIZE - (fft_data_size % PACKET_SIZE))

        freq_res = SAMPLING_FREQUENCY / fft_data_size
        fft_freq_axis = np.fft.fftfreq(fft_data_size, d=1/SAMPLING_FREQUENCY)[:fft_data_size//2]
        print(f"Achieved frequency resolution of {freq_res} Hz and a delay of {1000*fft_data_size/SAMPLING_FREQUENCY} ms.")
        fft_res_edit.setText("{:.2f}".format(freq_res))
    # Bind to a button.
    fft_set_button.clicked.connect(set_new_fft_data_size)
    
    # Ensure it's run before update_plot starts.
    set_new_fft_data_size()
    
    

    # Create a widget to hold the layout
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.addWidget(address_edit)
    layout.addWidget(port_edit)
    layout.addWidget(fft_res_label)
    layout.addWidget(fft_res_edit)
    layout.addWidget(fft_set_button)
    layout.addWidget(connect_button)
    
    def on_connection_finished():
        stop_event.clear()
        try:
            connect_button.setText("Connect") # This will fail if the GUI was closed.
        except:
            pass

    def connection_toggle():
        global socket_thread
        if socket_thread and socket_thread.is_alive():
            stop_event.set()
            socket_thread.join()
            socket_thread = None

        elif not socket_thread or not socket_thread.is_alive():
            port = int(port_edit.text())
            address = address_edit.text()
            connect_button.setText("Disconnect")
            socket_thread = threading.Thread(target=socket_connection, args=(address, port, data_queue, stop_event, on_connection_finished))
            socket_thread.start()

    # Bind the connection toggle to a button.
    connect_button.clicked.connect(connection_toggle)

    # Create a QWidgetItem to hold the QWidget
    proxy = QtWidgets.QGraphicsProxyWidget()
    proxy.setWidget(widget)

    # Add the proxy widget to the main window layout
    win_layout = win.ci.layout
    win_layout.addItem(proxy, 0, 0)

        # Define graph channel colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']

    # Create plots for 8 channels
    plots = [win.addPlot(row=i+1, col=0) for i in range(8)]
    curves = [plot.plot(pen=pg.mkPen(color)) for plot, color in zip(plots, colors)]  # Create curves to update

    # Create a single plot for FFT data
    fft_plot = win.addPlot(row=9, col=0)  # Common plot for FFT data
    fft_plot.setLabel("bottom", "Frequency", units="Hz")

    # Initialize a list of PlotDataItems for FFT curves
    fft_curves = [fft_plot.plot(pen=pg.mkPen(color)) for color in colors]

    # Set relative height of rows, giving more space to the last row
    win.ci.layout.setRowStretchFactor(9, 3)  # Stretch factor for the FFT plot
    for i in range(8):
        win.ci.layout.setRowStretchFactor(i, 1)  # Lower stretch factor for other plots

    # Calculate the stored data size for displaying on the plot
    samples_to_display = int(SAMPLING_FREQUENCY * args.time_frame)

    # Deques to store data for plotting
    channel_data = [collections.deque(maxlen=max(samples_to_display, fft_data_size)) for _ in range(8)]
    time_data = collections.deque(maxlen=samples_to_display)

    # Queue to share data between threads
    data_queue = queue.Queue()

    # Signal to stop the data-gathering thread
    stop_event = threading.Event()


    # Define the lambda function to update plots
    _update_plot = lambda: collect_data(data_queue, curves, fft_curves, time_data, channel_data, samples_to_display, fft_data_size, fft_freq_axis)

    # Set up a timer to refresh the plot at a regular interval
    timer = QtCore.QTimer()
    timer.timeout.connect(_update_plot)  # Connect the timer to the update function
    refresh_rate = int(1000*PACKET_SIZE/SAMPLING_FREQUENCY)
    print("Plot refresh rate:", refresh_rate)
    timer.start(refresh_rate)  # Update every time a new packet should be ready

    # Stop data-gathering thread when window is closed
    win.closeEvent = lambda _: stop_event.set()

    # Start the PyQt application
    # QtWidgets.QApplication.instance().exec_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
