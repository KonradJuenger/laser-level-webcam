from __future__ import annotations

from typing import Any

import numpy as np
import qimage2ndarray
from PySide6.QtCore import QObject
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot
from PySide6.QtGui import QImage
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QTransform
from PySide6.QtMultimedia import QVideoFrame

from src.curves import fit_gaussian
from src.DataClasses import FrameData
from src.utils import get_units


class SampleWorker(QObject):  # type: ignore
    """
    A worker class to process a stream of samples and emit the calculated mean.

    Attributes:
        OnSampleReady (Signal): Signal emitted when a sample is processed and a new mean is calculated.
        OnSubsampleRecieved (Signal): Signal emitted when a new subsample is received and processed.

    Methods:
        sample_in: Process a new subsample.
        start: Start the worker with a given number of total samples and outlier percentage to remove.
    """

    OnSampleReady = Signal(float)
    OnSubsampleRecieved = Signal(int)

    def __init__(self) -> None:
        super().__init__(None)
        self.ready = True
        self.sample_array = np.empty((0,), dtype=np.float64)
        self.total_samples = 0
        self.running_total = 0
        self.outlier_percent = 0.0
        self.started = False
        self._write_index = 0

    def sample_in(self, sample: float) -> None:
        """
        Process a new subsample by storing it in the buffer and emitting OnSubsampleRecieved.

        When the total number of subsamples is reached, the worker calculates the mean, removes the outlier
        percentage, and emits OnSampleReady with the new mean.

        Args:
            sample (float): A new subsample to process.
        """
        if not self.started or self.total_samples <= 0:
            return

        if self._write_index >= self.total_samples:
            return

        self.sample_array[self._write_index] = sample
        self._write_index += 1
        self.running_total += 1

        self.OnSubsampleRecieved.emit(self.running_total)

        if self.running_total == self.total_samples:
            n_outliers = int(self.total_samples * self.outlier_percent / 2.0)

            sorted_samples = np.sort(self.sample_array[: self.total_samples])
            if n_outliers:
                trimmed = sorted_samples[n_outliers : self.total_samples - n_outliers]
                if trimmed.size == 0:
                    trimmed = sorted_samples
            else:
                trimmed = sorted_samples

            mean = float(trimmed.mean())
            self.OnSampleReady.emit(mean)

            self.sample_array = np.empty((0,), dtype=np.float64)
            self.running_total = 0
            self.total_samples = 0
            self.started = False
            self._write_index = 0

    def start(self, total_samples: int, outlier_percent: float) -> None:
        """
        Start the worker with a given number of total samples and outlier percentage to remove.

        Args:
            total_samples (int): The total number of subsamples to process before emitting the mean.
            outlier_percent (float): The percentage of outliers to remove from the subsamples (0-100).
        """
        self.total_samples = total_samples
        self.outlier_percent = outlier_percent / 100.0
        self.running_total = 0
        self._write_index = 0

        if total_samples <= 0:
            self.sample_array = np.empty((0,), dtype=np.float64)
            self.started = False
            return

        self.sample_array = np.empty(total_samples, dtype=np.float64)
        self.started = True



class FrameWorker(QObject):  # type: ignore
    """
    A worker class to process a QVideoFrame and emit the corresponding image data.

    Attributes:
        OnFrameChanged (Signal): Signal emitted when the processed image data is ready.

    Methods:
        setVideoFrame(frame: QVideoFrame) -> None:
            Process a new QVideoFrame and emit the corresponding image data.

    """

    OnFrameChanged = Signal(list)
    OnCentreChanged = Signal(int)
    OnPixmapChanged = Signal(QPixmap)
    OnAnalyserUpdate = Signal(FrameData)

    def __init__(self, parent_obj: Any):
        super().__init__(None)
        self.ready = True
        self.analyser_smoothing = 0
        self._kernel = np.ones(1, dtype=np.float64)
        self._scope_levels = np.arange(256, dtype=np.uint16)
        self.centre = 0.0
        self.analyser_widget_height = 0
        self.parent_obj = parent_obj
        self.data_width = 0
        self.set_analyser_smoothing(0)

    @Slot(int)
    def set_analyser_smoothing(self, smoothing: int) -> None:
        radius = max(0, int(smoothing))
        size = 2 * radius + 1
        self.analyser_smoothing = radius
        self._kernel = np.ones(size, dtype=np.float64) / size

    @Slot(QVideoFrame)  # type: ignore
    def setVideoFrame(self, frame: QVideoFrame) -> None:
        """
        Process a new QVideoFrame and emit the corresponding image data.

        Args:
            frame (QVideoFrame): A QVideoFrame object to be processed.

        Returns:
            None

        """
        self.ready = False

        image = frame.toImage().convertToFormat(QImage.Format_Grayscale8)
        try:
            histo = np.mean(qimage2ndarray.raw_view(image), axis=0)
        except ValueError as e:
            print("Invalid QImage:", e)
            self.ready = True
            return

        pixmap = QPixmap.fromImage(image).transformed(QTransform().rotate(-90))
        self.OnPixmapChanged.emit(pixmap)

        histo = np.convolve(histo, self._kernel, mode="valid")
        histo = np.nan_to_num(histo)

        min_value, max_value = histo.min(), histo.max()
        if max_value > min_value:
            scaled = ((histo - min_value) * (255.0 / (max_value - min_value))).clip(0, 255)
        else:
            scaled = np.zeros_like(histo)

        histo_uint8 = scaled.astype(np.uint8)
        self.histo = histo_uint8

        histo_levels = histo_uint8.astype(np.uint16)
        scope_mask = self._scope_levels[None, :] < histo_levels[:, None]
        scopeData = scope_mask.astype(np.uint8) * 128

        qimage = QImage(
            scopeData.data,
            scopeData.shape[1],
            scopeData.shape[0],
            scopeData.strides[0],
            QImage.Format_Grayscale8,
        )

        a_pix = QPixmap.fromImage(qimage)
        a_pix = a_pix.transformed(QTransform().scale(1, -1))

        width = self.histo.shape[0]
        self.data_width = width

        a_sample = 0
        self.centre = fit_gaussian(self.histo)  # Specify the y position of the line
        self.OnCentreChanged.emit(self.centre)
        if self.centre:
            a_sample = int(self.analyser_widget_height - self.centre * self.analyser_widget_height / width)

        a_zero, a_text = 0, ""
        if self.parent_obj.zero and self.centre:  # If we have zero, we can set it and the text
            a_zero = int(self.analyser_widget_height - self.parent_obj.zero * self.analyser_widget_height / width)
            centre_real = (self.parent_obj.sensor_width / width) * (self.centre - self.parent_obj.zero)
            a_text = get_units(self.parent_obj.units, centre_real)

        frame_data = FrameData(a_pix, a_sample, a_zero, a_text)
        self.OnAnalyserUpdate.emit(frame_data)

        self.ready = True



class FrameSender(QObject):  # type: ignore
    """
    A class to send QVideoFrames.

    Attributes:
        OnFrameChanged (Signal): Signal emitted when a new QVideoFrame is ready to be processed.
    """

    OnFrameChanged = Signal(QVideoFrame)
