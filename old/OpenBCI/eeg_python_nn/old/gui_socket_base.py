import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
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

        self.plot_button = QtWidgets.QPushButton("Start Live Plot")
        self.plot_button.clicked.connect(self.start_live_plot)

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
        vbox.addWidget(self.plot_button)
        self.setLayout(vbox)

        self.data_thread = None
        self.data_queue = queue.Queue()

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

    def start_live_plot(self):
        # Start live plot
        pass

    def on_thread_finished(self):
        stop_event.clear()
        self.connect_button.setText("Connect")

    def closeEvent(self, event):
        self.disconnect_from_server()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = LivePlotApp()
    window.show()

    sys.exit(app.exec_())
