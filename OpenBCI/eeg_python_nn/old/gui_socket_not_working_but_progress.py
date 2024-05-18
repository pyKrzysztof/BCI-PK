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

# Event to signal when to stop the data gathering thread
stop_event = threading.Event()

# Data-gathering thread function
def gather_data(port, address, data_queue, callback):
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
        packets = socket.recv()
        if len(packets) == 0:
            if time.time() - last_packet_time > threshold:
                socket.shutdown(2)
                break
            continue

        data_queue.put(packets)
        last_packet_time = time.time()

    print("Closing connection.")
    socket.close()
    callback()


class LivePlotApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.port = 9999
        self.address = "localhost"
        self.fft_resolution = 3.33

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("EEG Data")

        self.port_label = QtWidgets.QLabel("Port:")
        self.port_text = QtWidgets.QLineEdit(str(self.port))
        self.address_label = QtWidgets.QLabel("Address:")
        self.address_text = QtWidgets.QLineEdit(self.address)
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)

        self.marker_checkbox = QtWidgets.QCheckBox("Receive Marker Data")
        self.fft_checkbox = QtWidgets.QCheckBox("Calculate FFT Data")
        self.fft_res_label = QtWidgets.QLabel("FFT Resolution:")
        self.fft_res_text = QtWidgets.QLineEdit(str(self.fft_resolution))

        self.live_plot_checkbox = QtWidgets.QCheckBox("Show Live Plot")
        self.live_plot_checkbox.stateChanged.connect(self.toggle_live_plot)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.port_label)
        vbox.addWidget(self.port_text)
        vbox.addWidget(self.address_label)
        vbox.addWidget(self.address_text)
        vbox.addWidget(self.connect_button)
        vbox.addWidget(self.marker_checkbox)
        vbox.addWidget(self.fft_checkbox)
        vbox.addWidget(self.fft_res_label)
        vbox.addWidget(self.fft_res_text)
        vbox.addWidget(self.live_plot_checkbox)
        self.setLayout(vbox)

        self.data_thread = None
        self.data_queue = queue.Queue()

        self.curves = []
        self.fft_curves = []

        self.win = None
        self.fft_plot = None
    
    def toggle_connection(self):
        if not self.data_thread or not self.data_thread.is_alive():
            self.connect_to_server()
        else:
            self.disconnect_from_server()

    def connect_to_server(self):
        self.port = int(self.port_text.text())
        self.address = self.address_text.text()
        self.fft_resolution = float(self.fft_res_text.text())

        if not self.data_thread or not self.data_thread.is_alive():
            self.data_thread = threading.Thread(target=gather_data, args=(self.port, self.address, self.data_queue, self.on_thread_finished))
            self.data_thread.start()
            self.connect_button.setText("Disconnect")

    def disconnect_from_server(self):
        if self.data_thread and self.data_thread.is_alive():
            stop_event.set()
            self.data_thread.join()
            self.connect_button.setText("Connect")

    def on_thread_finished(self):
        stop_event.clear()
        self.connect_button.setText("Connect")

    def toggle_live_plot(self):
        if self.live_plot_checkbox.isChecked():
            self.create_plots()
        else:
            if self.win is not None:
                self.win.close()
                self.win = None

    def create_plots(self):
        if self.win is None:
            self.win = pg.GraphicsLayoutWidget(show=True)
            self.win.setWindowTitle("EEG Data")

            # Define graph channel colors
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']

            # Create plots for 8 channels
            plots = [self.win.addPlot(row=i, col=0) for i in range(8)]
            self.curves = [plot.plot(pen=pg.mkPen(color)) for plot, color in zip(plots, colors)]  # Create curves to update

            # Create a single plot for FFT data
            self.fft_plot = self.win.addPlot(row=8, col=0)  # Common plot for FFT data
            self.fft_plot.setLabel("bottom", "Frequency", units="Hz")

            fft_data_size = int(250 // self.fft_resolution)
            if fft_data_size % 25 != 0:
                fft_data_size = int(fft_data_size + 25 - (fft_data_size % 25))

            fft_res = 250 / fft_data_size
            fft_freq_axis = np.fft.fftfreq(fft_data_size, d=1/250)[:fft_data_size//2]
            print(f"Achieved frequency resolution of {fft_res} Hz and a delay of {1000*fft_data_size/250} ms.")

            # Initialize a list of PlotDataItems for FFT curves
            self.fft_curves = [self.fft_plot.plot(pen=pg.mkPen(color)) for color in colors]

            # Set relative height of rows, giving more space to the last row
            self.win.ci.layout.setRowStretchFactor(8, 3)  # Stretch factor for the FFT plot
            for i in range(8):
                self.win.ci.layout.setRowStretchFactor(i, 1)  # Lower stretch factor for other plots

            # Calculate the stored data size for displaying on the plot
            samples_to_display = int(250 * 3.0)

            # Deques to store data for plotting
            channel_data = [collections.deque(maxlen=max(samples_to_display, fft_data_size)) for _ in range(8)]
            time_data = collections.deque(maxlen=samples_to_display)

            # Define the lambda function to update plots
            _update_plot = lambda: self.update_plot(self.data_queue, self.curves, self.fft_curves, time_data, channel_data, samples_to_display, fft_data_size, fft_freq_axis)

            # Set up a timer to refresh the plot at a regular interval
            timer = QtCore.QTimer()
            timer.timeout.connect(_update_plot)  # Connect the timer to the update function
            timer.start(100)  # Update every 100 ms

            # Stop data-gathering thread when window is closed
            self.win.closeEvent = lambda _: stop_event.set()

    def update_plot(self, data_queue, curves, fft_curves, time_data, channel_data, samples_to_display, fft_data_size, fft_freq_axis):
        print("test lol")
        try:
            # Get data from the queue
            data = data_queue.get(timeout=0.2)
            print(data)
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
            print("empty")
            pass

    def closeEvent(self, event):
        self.disconnect_from_server()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = LivePlotApp()
    window.show()

    sys.exit(app.exec_())
