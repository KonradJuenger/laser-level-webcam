from __future__ import annotations

import threading
from typing import List

import cv2
import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtCore import QThread
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot
from PySide6.QtMultimedia import QMediaDevices
from scipy.stats import linregress

from src.DataClasses import Sample
from src.Workers import FrameSender
from src.Workers import FrameWorker
from src.Workers import SampleWorker


def samples_recalc(samples: list[Sample]) -> None:
    """
    Recalculates the linear regression and errors of the given list of samples.

    Args:
    - samples (list): A list of Sample objects with x and y attributes.

    Returns:
    - None

    Example:
    - sample1 = Sample(1, 2)
      sample2 = Sample(2, 4)
      sample3 = Sample(3, 6)
      samples_recalc([sample1, sample2, sample3])
    """
    # Ensure that there are at least 3 samples to calculate the linear regression and errors.
    if len(samples) >= 3:
        # Get the x and y values from the samples.
        x = [s.x for s in samples]
        y = [s.y for s in samples]

        # Calculate the linear regression using the x and y values.
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Calculate the minimum and maximum y errors for each sample.
        minYError = float("inf")
        maxYError = float("-inf")
        for s in samples:
            s.linYError = s.y - (slope * s.x + intercept)
            if s.linYError > maxYError:
                maxYError = s.linYError
            if s.linYError < minYError:
                minYError = s.linYError

        # Calculate the shim and scrape values for each sample.
        for s in samples:
            # Make highest point zero for shimming, we are going to shim up all the low points to this height.
            s.shim = maxYError - s.linYError
            # Make lowest point zero for scraping, we are going to scrape off all the high areas.
            s.scrape = s.linYError - minYError


class OpenCVCaptureThread(QThread):  # type: ignore
    frameCaptured = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending_index: int | None = None
        self._should_stop = False
        self._capture: cv2.VideoCapture | None = None

    def start_capture(self, index: int) -> None:
        with self._lock:
            self._pending_index = index
            self._should_stop = False
        if not self.isRunning():
            self.start()

    def set_device(self, index: int) -> None:
        self.start_capture(index)

    def stop(self) -> None:
        with self._lock:
            self._should_stop = True
        self.wait()

    def run(self) -> None:  # noqa: D401 - QThread run loop
        while True:
            with self._lock:
                if self._should_stop:
                    break
                pending = self._pending_index
                self._pending_index = None
            if pending is not None:
                self._open_capture(pending)
            capture = self._capture
            if capture is None:
                self.msleep(100)
                continue
            ok, frame = capture.read()
            if not ok:
                self.msleep(10)
                continue
            self.frameCaptured.emit(frame)
        self._release_capture()
        with self._lock:
            self._should_stop = False
            self._pending_index = None

    def _open_capture(self, index: int) -> None:
        self._release_capture()
        capture = self._create_capture(index)
        if capture is None:
            return
        self._capture = capture

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @staticmethod
    def _create_capture(index: int) -> cv2.VideoCapture | None:
        backends: list[int] = []
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
            cap.release()
        return None


class Core(QObject):  # type: ignore
    OnAnalyserUpdate = Signal(list)
    OnSubsampleProgressUpdate = Signal(list)
    OnSampleComplete = Signal()
    OnUnitsChanged = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.centre = 0.0  # The found centre of the histogram
        self.zero = 0.0  # The zero point
        self.analyser_widget_height = 0  # The height of the widget so we can calculate the offset
        self.subsamples = 0  # total number of subsamples
        self.outliers = 0  # percentage value of how many outliers to remove from a sample
        self.units = ""  # string representing the units
        self.sensor_width = 0  # width of the sensor in millimeters (mm)
        self.setting_zero_sample = False  # boolean if we are setting zero or a sample
        self.replacing_sample = False  # If we are replacing a sample
        self.replacing_sample_index = 0  # the index of the sample we are replacing
        self.line_data = np.empty(0)  # numpy array of the fitted line through the samples
        self.samples: list[Sample] = []
        self.shutting_down = False

        # Frame worker
        self.workerThread = QThread()
        self.frameSender = FrameSender()
        self.frameWorker = FrameWorker(parent_obj=self)
        self.frameWorker.moveToThread(self.workerThread)
        self.workerThread.start()
        self.frameSender.OnFrameChanged.connect(self.frameWorker.process_frame)

        # Capture thread
        self.capture_thread = OpenCVCaptureThread()
        self.capture_thread.frameCaptured.connect(self.on_frame_captured)

        # Sample worker
        self.sample_worker = SampleWorker()
        self.sample_worker.OnSampleReady.connect(self.received_sample)
        self.sample_worker.OnSubsampleRecieved.connect(self.subsample_progress_update)
        self.sampleWorkerThread = QThread()
        self.sample_worker.moveToThread(self.sampleWorkerThread)
        self.sampleWorkerThread.start()

        self._camera_indices = self._enumerate_cameras()
        if not self._camera_indices:
            self._camera_indices = [0]
        self._camera_names = self._resolve_camera_names(self._camera_indices)
        self.capture_thread.start_capture(self._camera_indices[0])

    def delete_sample(self, index: int) -> None:
        debug = False
        if debug:
            print(f"num samples = {len(self.samples)}")
            print(f"Before: {self.samples=}")

        del self.samples[index]

        # Fix the indexes
        for index, sample in enumerate(self.samples):
            sample.x = index

        if debug:
            print(f"After: {self.samples=}")

    def subsample_progress_update(self, subsample: Sample) -> None:
        self.OnSubsampleProgressUpdate.emit([subsample, self.subsamples])  # current sample and total

    def received_sample(self, val: float) -> None:
        if self.setting_zero_sample:
            self.zero = val
        else:
            size_in_mm = (self.sensor_width / self.frameWorker.data_width) * (val - self.zero)

            if self.replacing_sample:
                x_orig = self.samples[self.replacing_sample_index].x
                self.samples[self.replacing_sample_index] = Sample(x=x_orig, y=size_in_mm)
                self.replacing_sample = False

            else:  # Append to samples
                self.samples.append(Sample(x=len(self.samples), y=size_in_mm))

            samples_recalc(self.samples)

        self.OnSampleComplete.emit()

    def set_units(self, units: str) -> None:
        self.units = units

        self.OnUnitsChanged.emit(self.units)

    def start_sample(self, zero: bool, replacing_sample: bool, replacing_sample_index: int) -> None:
        self.replacing_sample = replacing_sample
        self.replacing_sample_index = replacing_sample_index

        if zero:  # if we are zero, we reset everything
            self.line_data = np.empty(0)
            self.zero = 0.0

        self.setting_zero_sample = zero
        self.sample_worker.start(self.subsamples, self.outliers)

    @Slot(object)  # type: ignore[arg-type]
    def on_frame_captured(self, frame: np.ndarray) -> None:
        if self.frameWorker.ready:
            self.frameSender.OnFrameChanged.emit(frame)

    def get_cameras(self) -> list[str]:
        return self._camera_names.copy()

    def set_camera(self, index: int) -> None:
        if index < 0 or index >= len(self._camera_indices):
            return
        device_index = self._camera_indices[index]
        self.capture_thread.set_device(device_index)

    def shutdown(self) -> None:
        """Stop capture and worker threads safely to allow application exit."""
        if self.shutting_down:
            return

        self.shutting_down = True

        try:
            self.capture_thread.frameCaptured.disconnect(self.on_frame_captured)
        except Exception:
            pass
        self.capture_thread.stop()

        try:
            self.frameSender.OnFrameChanged.disconnect(self.frameWorker.process_frame)
        except Exception:
            pass

        self.workerThread.requestInterruption()
        self.workerThread.quit()
        if not self.workerThread.wait(2000):
            self.workerThread.terminate()
            self.workerThread.wait()

        self.sampleWorkerThread.quit()
        if not self.sampleWorkerThread.wait(2000):
            self.sampleWorkerThread.terminate()
            self.sampleWorkerThread.wait()

        self.shutting_down = False

    def _enumerate_cameras(self, max_devices: int = 5) -> List[int]:
        found: List[int] = []
        for index in range(max_devices):
            capture = OpenCVCaptureThread._create_capture(index)
            if capture is not None:
                found.append(index)
                capture.release()
        return found

    def _resolve_camera_names(self, indices: List[int]) -> List[str]:
        qt_devices = list(QMediaDevices.videoInputs())
        if not qt_devices:
            return [f"Camera {idx}" for idx in indices]
        names: List[str] = []
        for pos, idx in enumerate(indices):
            if pos < len(qt_devices):
                names.append(qt_devices[pos].description())
            else:
                names.append(f"Camera {idx}")
        return names
