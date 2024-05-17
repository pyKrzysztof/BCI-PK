import logging
import argparse
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from my_utils import calculate_time
import sys

class AppBase:

    def __init__(self, board_shim: BoardShim=None, from_file="", update_speed_ms=50, points_to_analyze=32, display_s=4):
        self.board = board_shim
        self.source_file = from_file
        self.file_index = 0
        self.data = None

        self.sampling_rate = BoardShim.get_sampling_rate(board_id=BoardIds.CYTON_BOARD)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id=BoardIds.CYTON_BOARD)
        self.marker_channel = BoardShim.get_marker_channel(board_id=BoardIds.CYTON_BOARD)
        self.update_speed_ms = update_speed_ms
        self.window_size = 4
        self.time_to_display = display_s
        self.points_to_display = self.time_to_display * self.sampling_rate
        self.points_to_analyze = points_to_analyze

        if from_file != "":
            self.data = DataFilter.read_file(from_file)
            self.get_data_packet = self.get_data_packet_file
        else:
            self.get_data_packet = self.get_data_packet_live
        
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True)

        self.init_pens()
        self.init_ui()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)

        QtWidgets.QApplication.instance().exec()

    def get_data_packet_live(self):
        if self.board.get_board_data_count() >= self.points_to_analyze:
            return self.board.get_board_data()
        return None

    def get_data_packet_file(self) -> np.ndarray:
        try:
            indices = list(range(0, self.points_to_analyze, 1))
            data = np.array(self.data[:, indices], order="C")
            self.data = np.delete(self.data, indices, axis=1)
        except IndexError:
            if len(self.data) > 0:
                data = np.array(self.data, order="C")
                self.data = None
        except:
            data = None
        finally:
            return data

    def init_pens(self):
        self.pens = list()
        self.default_pen = pg.mkPen({"color": "#000", "width": 1})
        self.brushes = list()
        colors = ['#A54E4E', '#A473B6', '#5B45A4', '#2079D2', '#32B798', '#2FA537', '#9DA52F', '#A57E2F', '#A53B2F']
        for i in range(len(colors)):
            pen = pg.mkPen({'color': colors[i], 'width': 2})
            self.pens.append(pen)
            brush = pg.mkBrush(colors[i])
            self.brushes.append(brush)

    def init_ui(self):
        pass

    def update(self):
        pass






class AppAnalyzer(AppBase):

    def __init__(self, board_shim: BoardShim=None, from_file="", update_speed_ms=50, points_to_analyze=32, display_s=4, processes=[], psd_size=256):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        rows = 24 if board_shim is None or board_shim.board_id == BoardIds.CYTON_BOARD else 32
        self.full_data = []
        self.processes = processes
        self.psd_size = psd_size

        for process in processes:
            self.full_data.append(np.zeros((rows, 1), order="C"))
        super().__init__(board_shim = board_shim, from_file = from_file, update_speed_ms = update_speed_ms, points_to_analyze = points_to_analyze, display_s = display_s)

    def init_ui(self):
        self.init_timeseries()
        self.init_psd()

    def init_timeseries(self):
        self.plots = list()
        self.curves = [[] for _ in range(len(self.processes))]
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            self.plots.append(p)

            for idx in range(len(self.processes)):
                if idx == 0:
                    curve = p.plot(pen=self.pens[i % len(self.pens)])
                else:
                    curve = p.plot(pen=self.default_pen)
                    # curve.setDownsampling(auto=True, method='mean', ds=3)
                self.curves[idx].append(curve)

    def init_psd(self):
        self.psd_plots = []
        self.psd_curves = [[] for _ in range(len(self.processes))]
        for i in range(len(self.processes)):
            plot = self.win.addPlot(row=i if i == 0 else i+(len(self.eeg_channels) // 2), col=1, rowspan=len(self.eeg_channels) // 2)
            # plot.showAxis("left", True)
            plot.setTitle(f"FFT {i+1}")
            plot.setMenuEnabled('left', False)
            # plot.setLogMode(False, True)

            for j in range(len(self.eeg_channels)):
                curve = plot.plot(pen=self.pens[j%len(self.pens)])
                curve.setDownsampling(auto=True, method='mean', ds=3)
                self.psd_curves[i].append(curve)
            
            self.psd_plots.append(plot)


    def update(self):
        data = self.get_data_packet()
        if data is None:
            self.app.processEvents()
            return

        data_copies = [np.copy(data, order="C") for _ in range(len(self.processes))]
        
        
        # process the data
        for i, process in enumerate(self.processes):
            for count, channel in enumerate(self.eeg_channels):
                data_copies[i] = process(data_copies[i], channel, self.sampling_rate)

            # join packet data to the full data for each process       
            self.full_data[i] = np.array(np.concatenate((self.full_data[i], data_copies[i]), axis=1), order="C")
        
        # plot timeseries
        for count, channel in enumerate(self.eeg_channels):
            for curves_ts, curves_psd, display_data in zip(self.curves, self.psd_curves, self.full_data):
                curves_ts[count].setData(display_data[channel][(-min(len(display_data[channel]), self.points_to_display)):])

        # calculate and plot fft data
        for i in range(len(self.processes)):
            for count, channel in enumerate(self.eeg_channels):
                if self.full_data[i].shape[1] > self.psd_size:
                    psd_data = DataFilter.get_psd_welch(self.full_data[i][channel][-self.psd_size:], self.psd_size, self.psd_size // 2,
                                                        self.sampling_rate,
                                                        WindowOperations.HAMMING.value)
                    lim = min(70, len(psd_data[0]))
                    self.psd_curves[i][count].setData(psd_data[1][0:lim].tolist(), psd_data[0][0:lim].tolist())


        self.app.processEvents()


# for using with file playback, you need to comment out row-major check in brainflow.utils.py file.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", required=False)
    parser.add_argument("--from_file", type=str, default="", required=False)
    parser.add_argument("--app_id", type=int, default=0, required=False, help="Choose application to use, 0 for live data analysis, 1 for comparison of data between operations.")
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    
    params = BrainFlowInputParams()
    params.serial_port = args.port


    def process_1(data, channel, sampling_rate):
        DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data[channel], sampling_rate, 1.0, 46.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0) # PERFORMS POORLY ON (AT THE VERY LEAST ON ARTIFICIAL DATA)
        DataFilter.perform_bandstop(data[channel], sampling_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        return data

    def process_2(data, channel, sampling_rate):
        DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
        # DataFilter.perform_bandpass(data[channel], sampling_rate, 2.0, 45.0, 2,
        #                             FilterTypes.BUTTERWORTH_ZERO_PHASE, 0) # PERFORMS POORLY ON (AT THE VERY LEAST ON ARTIFICIAL DATA)
        # DataFilter.perform_bandstop(data[channel], sampling_rate, 48.0, 52.0, 2,
        #                                     FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        # DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 2,
        #                             FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        return data

    app_handle = AppAnalyzer if args.app_id == 0 else None
    kwargs = {"board_shim": None, "from_file": args.from_file, "points_to_analyze": 64, 
              "update_speed_ms": 100, "display_s": 2, "processes": [process_1, process_2], "psd_size": 128}

    if args.from_file != "":
        app = app_handle(**kwargs)
    else:
        board = BoardShim(board_id=BoardIds.SYNTHETIC_BOARD, input_params=params)
        board.prepare_session()
        board.start_stream(45000)
        kwargs["board_shim"] = board
        app = app_handle(**kwargs)
    try:
        board.stop_stream()
        board.release_session()
    except:
        pass

if __name__ == "__main__":
    main()

# low 