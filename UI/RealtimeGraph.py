import argparse
import logging
import sys  # We need sys so that we can pass argv to QApplication
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(
            title='BrainFlow Plot', size=(800, 600))
        # self.win.setBackground('w')
        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        self.win.show()
        sys.exit(self.app.exec_())

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        self.labels = ['FP1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5', 'P3',
                       'FP2', 'FC6', 'FT8', 'C4', 'F8', 'T8', 'P7', 'P8']
        self.colors = ['gray', 'purple', 'cyan', 'green', 'yellow', 'orange', 'red', 'brown',
                       'gray', 'purple', 'cyan', 'green', 'yellow', 'orange', 'red', 'brown']
        for i in range(len(self.exg_channels)):

            p = self.win.addPlot(row=i, col=0, pen='blue')
            p.showAxis('bottom', False)
            p.setLabel("left", self.labels[i])
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot(pen=self.colors[i])
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_wavelet_denoising(data[channel], WaveletTypes.BIOR3_9, 3, WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
            #                              WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM8')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.CYTON_DAISY_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port


    try:
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()