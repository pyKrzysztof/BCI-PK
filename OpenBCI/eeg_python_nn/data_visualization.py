import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations


class AppBase:

    window_method = WindowOperations.BLACKMAN_HARRIS.value

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
            self.file_data = DataFilter.read_file(from_file)
            self.data = np.array(np.copy(self.file_data), order="C")
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

    # def start(self):

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
        self.pens = []
        self.default_pen = pg.mkPen({"color": "#000", "width": 1})
        self.brushes = []
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

    def __init__(self, board_shim: BoardShim=None, from_file="", update_speed_ms=50, points_to_analyze=32, display_s=4, processes=[], fft_size=256):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        rows = 24 if board_shim is None or board_shim.board_id == BoardIds.CYTON_BOARD else 32
        self.full_data = []
        self.processes = processes
        self.fft_size = fft_size

        for process in processes:
            self.full_data.append(np.zeros((rows, 1), order="C"))
        super().__init__(board_shim = board_shim, from_file = from_file, update_speed_ms = update_speed_ms, points_to_analyze = points_to_analyze, display_s = display_s)

    def init_ui(self):
        self.init_timeseries()
        self.init_fft()

    def init_timeseries(self):
        self.plots = []
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

    def init_fft(self):
        self.fft_plots = []
        self.fft_curves = [[] for _ in range(len(self.processes))]
        increment = len(self.eeg_channels) // len(self.processes)
        for i in range(len(self.processes)):
            plot = self.win.addPlot(row=i if i == 0 else i*increment, col=1, rowspan=increment)
            # plot.showAxis("left", True)
            plot.setTitle(f"FFT process no.{i+1}")
            plot.setMenuEnabled('left', False)
            # plot.setLogMode(False, True)

            for j in range(len(self.eeg_channels)):
                curve = plot.plot(pen=self.pens[j%len(self.pens)])
                curve.setDownsampling(auto=True, method='mean', ds=3)
                self.fft_curves[i].append(curve)
            
            self.fft_plots.append(plot)


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
            for curves_ts, curves_fft, display_data in zip(self.curves, self.fft_curves, self.full_data):
                curves_ts[count].setData(display_data[channel][(-min(len(display_data[channel]), self.points_to_display)):])

        # calculate and plot fft data
        for i in range(len(self.processes)):
            for count, channel in enumerate(self.eeg_channels):
                if self.full_data[i].shape[1] > self.fft_size:
                    fft_data = DataFilter.get_psd_welch(self.full_data[i][channel][-self.fft_size:], self.fft_size, self.fft_size // 2,
                                                        self.sampling_rate,
                                                        AppBase.window_method)
                    lim = min(70, len(fft_data[0]))
                    self.fft_curves[i][count].setData(fft_data[1][0:lim].tolist(), fft_data[0][0:lim].tolist())


        self.app.processEvents()

class AppPresenter(AppAnalyzer):

    def init_ui(self):
        if self.source_file != "":
            self.init_controls()
            self.update = self.update_file
        else:
            self.update = self.update_live
        super().init_ui()

    def init_controls(self):
        label_1 = QtWidgets.QLabel("Min. packet:")
        label_2 = QtWidgets.QLabel("Duration (N):")
        label_3 = QtWidgets.QLabel("Playback speed:")
        self.packet_show_start = QtWidgets.QLineEdit("auto")
        self.packet_show_duration = QtWidgets.QLineEdit("1000")
        self.playback_speed = QtWidgets.QLineEdit("1.0")
        self.current_packet = QtWidgets.QLabel("0")
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.addWidget(label_1)
        layout.addWidget(self.packet_show_start)
        layout.addWidget(self.current_packet)
        layout.addWidget(label_2)
        layout.addWidget(self.packet_show_duration)
        layout.addWidget(label_3)
        layout.addWidget(self.playback_speed)
        layout.addWidget
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(widget)
        self.win.ci.layout.addItem(proxy, 0, 0)


    def init_timeseries(self):
        self.plots = []
        self.curves = []
        self.current_packet_index = 0
        for i in range(len(self.processes)):
            p = self.win.addPlot(row=i+1, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            self.plots.append(p)
            curve = p.plot(pen=self.pens[i % len(self.pens)])
            # curve.setDownsampling(auto=True, method='mean', ds=3)
            self.curves.append(curve)

    def init_fft(self):
        self.fft_plots = []
        self.fft_curves = []
        for i in range(len(self.processes)):
            plot = self.win.addPlot(row=i+1, col=1)
            plot.showAxis("left", True)
            plot.setTitle(f"FFT {i+1}")
            plot.setMenuEnabled('left', False)
            # plot.setLogMode(False, True)

            curve = plot.plot(pen=self.pens[i%len(self.pens)])
            # curve.setDownsampling(auto=True, method='mean', ds=3)
            self.fft_curves.append(curve)
            self.fft_plots.append(plot)
            if i == 0:
                continue
            plot.setYRange(0, 100, padding=5)
            plot.vb.setLimits(yMin=0, yMax=100)


    
    def update(self):
        pass

    # for file data
    # @calculate_time
    def update_file(self):
        data = self.get_data_packet()
        channel = self.eeg_channels[0]
        while data is not None:
            data_copies = [np.copy(data, order="C") for _ in range(len(self.processes))]
            # process the data
            for i, process in enumerate(self.processes):
                data_copies[i] = process(data_copies[i], channel, self.sampling_rate)

                # join packet data to the full data for each process       
                self.full_data[i] = np.array(np.concatenate((self.full_data[i], data_copies[i]), axis=1), order="C")

            data = self.get_data_packet()
        
        try:
            if self.packet_show_start != "auto":
                duration = int(self.packet_show_duration.text())
                idx_plot_start = int(self.packet_show_start.text())
                idx_plot_end = idx_plot_start + duration
                self.current_packet_index = idx_plot_start
            else: raise
        except:
            try:
                duration = int(self.packet_show_duration.text())
                idx_plot_start = self.current_packet_index
                idx_plot_end = idx_plot_start + duration
                self.current_packet_index = self.current_packet_index + int(float(self.playback_speed.text())*(self.update_speed_ms+30)*self.sampling_rate/1000) # 30ms is the average plotting delay for this function calculated using @calculate_time decorator
            except:
                self.app.processEvents()
                return

        if idx_plot_end >= self.full_data[0].shape[1]:
            self.current_packet_index = self.full_data[0].shape[1] - duration
            idx_plot_end = -1
            idx_plot_start = -duration
        self.current_packet.setText(f"{self.current_packet_index}")    

        # plot timeseries
        for i in range(len(self.processes)):
            self.curves[i].setData(self.full_data[i][channel][idx_plot_start:idx_plot_end])

        # # calculate and plot fft data
        for i in range(len(self.processes)):
            try:
                fft_data = DataFilter.get_psd_welch(self.full_data[i][channel][idx_plot_end-self.fft_size:idx_plot_end], self.fft_size, self.fft_size // 2,
                                                    self.sampling_rate,
                                                    AppBase.window_method)
                lim = min(70, len(fft_data[0]))
                self.fft_curves[i].setData(fft_data[1][0:lim].tolist(), fft_data[0][0:lim].tolist())
            except:
                pass

        self.app.processEvents()

    # for live data
    # @calculate_time
    def update_live(self):
        data = self.get_data_packet()
        if data is None:
            self.app.processEvents()
            return

        data_copies = [np.copy(data, order="C") for _ in range(len(self.processes))]
        channel = self.eeg_channels[0]

        # process the data
        for i, process in enumerate(self.processes):
            data_copies[i] = process(data_copies[i], channel, self.sampling_rate)

            # join packet data to the full data for each process       
            self.full_data[i] = np.array(np.concatenate((self.full_data[i], data_copies[i]), axis=1), order="C")

        idx_plot_start = -min(self.full_data[0].shape[1], self.points_to_display)
        idx_plot_end = -1

        # plot timeseries
        for i in range(len(self.processes)):
            self.curves[i].setData(self.full_data[i][channel][idx_plot_start:idx_plot_end])

        # calculate and plot fft data
        for i in range(len(self.processes)):
            if self.full_data[i].shape[1] > self.fft_size:
                fft_data = DataFilter.get_fft_welch(self.full_data[i][channel][-self.fft_size:], self.fft_size, self.fft_size // 2,
                                                    self.sampling_rate,
                                                    WindowOperations.HAMMING.value)
                lim = min(70, len(fft_data[0]))
                self.fft_curves[i].setData(fft_data[1][0:lim].tolist(), fft_data[0][0:lim].tolist())


        self.app.processEvents()
