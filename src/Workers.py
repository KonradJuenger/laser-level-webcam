from __future__ import annotations

from typing import Any
from collections import deque

import cv2
import numpy as np
import time
from PySide6.QtCore import QObject
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage
from PySide6.QtGui import QTransform
from pathlib import Path

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
    ENABLE_TIMING = False
    """
    A worker class to process camera frames represented as NumPy arrays and emit the corresponding image data.
    """

    TIMING_KEYS = ("convert", "channel_extract", "emit", "profile", "centre", "histogram")

    OnFrameChanged = Signal(list)
    OnCentreChanged = Signal(float)
    OnImageReady = Signal(QImage)
    OnAnalyserUpdate = Signal(FrameData)
    OnChannelsAvailable = Signal(list)

    previewEnabledChanged = Signal(bool)

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
        self.channel = "Intensity"
        self._available_channels: list[str] = ["Intensity"]
        self.measurement_mode = "Gaussian Peak"
        self.preview_enabled = True
        self.preview_skip = 0
        self.preview_interval = 0
        self.threshold_enabled = False
        self.threshold_value = 0.0
        self._timing_samples = {key: deque(maxlen=120) for key in self.TIMING_KEYS}
        self.padding_mode = 'zeros'
        self._timing_frames = 0
        self._timing_last_log = time.perf_counter()
        self._timing_base_enabled = bool(self.ENABLE_TIMING)
        self.logging_until = 0.0
        self.log_path = Path(__file__).resolve().parent / "performance.log"
        self.set_analyser_smoothing(0)

    @Slot(int)
    def set_analyser_smoothing(self, smoothing: int) -> None:
        radius = max(0, int(smoothing))
        size = 2 * radius + 1
        self.analyser_smoothing = radius
        self._kernel = np.ones(size, dtype=np.float64) / size

    @Slot(str)
    def set_channel(self, channel: str) -> None:
        self.channel = channel

    @Slot(str)
    def set_measurement_mode(self, mode: str) -> None:
        self.measurement_mode = mode

    @Slot(bool)
    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled = enabled
        self.preview_skip = 0
        self.previewEnabledChanged.emit(enabled)

    @Slot(bool)
    def set_threshold_enabled(self, enabled: bool) -> None:
        self.threshold_enabled = bool(enabled)

    @Slot(str)
    def set_padding_mode(self, mode: str) -> None:
        if mode in {'zeros', 'edge', 'reflect'}:
            self.padding_mode = mode
        else:
            self.padding_mode = 'zeros'

    @Slot(int)
    def set_threshold_value(self, value: int) -> None:
        clamped = max(0, min(255, int(value)))
        self.threshold_value = float(clamped)

    @Slot(object)  # type: ignore[arg-type]
    def process_frame(self, frame: np.ndarray) -> None:
        if getattr(self.parent_obj, "shutting_down", False) or QThread.currentThread().isInterruptionRequested():
            self.ready = True
            return

        if frame is None or frame.size == 0:
            self.ready = True
            return

        self.ready = False

        timing_active = self._timing_enabled()
        timing_last = time.perf_counter() if timing_active else None

        available_channels = ["Intensity"]
        if frame.ndim == 3 and frame.shape[2] >= 3:
            available_channels.extend(["Red", "Green", "Blue"])
        if available_channels != self._available_channels:
            self._available_channels = available_channels
            self.OnChannelsAvailable.emit(available_channels)
            if self.channel not in available_channels:
                self.channel = available_channels[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        if timing_active:
            timing_last = self._timing_checkpoint("convert", timing_last)

        update_preview = False
        if self.preview_enabled:
            update_preview = (self.preview_skip % 2 == 0)
            self.preview_skip = (self.preview_skip + 1) % 2
        else:
            self.preview_skip = 0

        preview_rgb = None
        if self.channel == "Intensity" or frame.ndim < 3:
            plane_source = gray.astype(np.float32, copy=False)
            preview_plane_u8 = np.ascontiguousarray(gray)
        else:
            if self.channel == "Red":
                channel_data = frame[:, :, 2]
            elif self.channel == "Green":
                channel_data = frame[:, :, 1]
            elif self.channel == "Blue":
                channel_data = frame[:, :, 0]
            else:
                channel_data = frame[:, :, :3].mean(axis=2)
            plane_source = channel_data.astype(np.float32, copy=False)
            preview_plane_u8 = np.ascontiguousarray(np.clip(plane_source, 0, 255).astype(np.uint8))

        if self.threshold_enabled and self.threshold_value > 0.0:
            threshold = self.threshold_value
            mask = plane_source >= threshold
            plane_source[...] = np.where(mask, 255.0, 0.0)
            preview_plane_u8[...] = np.where(mask, 255, 0).astype(np.uint8)

        if self.channel != "Intensity" and frame.ndim >= 3:
            height, width = preview_plane_u8.shape
            preview_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            if self.channel == "Red":
                preview_rgb[:, :, 0] = preview_plane_u8
            elif self.channel == "Green":
                preview_rgb[:, :, 1] = preview_plane_u8
            elif self.channel == "Blue":
                preview_rgb[:, :, 2] = preview_plane_u8
            else:
                preview_rgb[:] = preview_plane_u8[:, :, None]

        if update_preview:
            if preview_rgb is None:
                preview_image = self._to_qimage(preview_plane_u8)
            else:
                preview_image = self._to_qimage_rgb(preview_rgb)
            rotated_image = preview_image.transformed(QTransform().rotate(-90))
            self.OnImageReady.emit(rotated_image)
            if timing_active:
                timing_last = self._timing_checkpoint("emit", timing_last)

        plane = np.ascontiguousarray(plane_source)
        if timing_active:
            timing_last = self._timing_checkpoint("channel_extract", timing_last)

        plane_mean = plane.mean(axis=0)
        pad_width = plane_mean.size
        pad_mode = 'constant' if self.padding_mode == 'zeros' else ('edge' if self.padding_mode == 'edge' else 'reflect')
        pad_kwargs = {'constant_values': 0.0} if pad_mode == 'constant' else {}
        padded = np.pad(plane_mean, pad_width, mode=pad_mode, **pad_kwargs)
        histo = np.convolve(padded, self._kernel, mode="same")
        histo = histo[pad_width:pad_width + plane_mean.size]
        histo = np.nan_to_num(histo)
        self.histo = histo

        if timing_active:
            timing_last = self._timing_checkpoint("profile", timing_last)

        min_value, max_value = histo.min(), histo.max()
        if max_value > min_value:
            scaled = ((histo - min_value) * (255.0 / (max_value - min_value))).clip(0, 255)
        else:
            scaled = np.zeros_like(histo)

        histo_uint8 = scaled.astype(np.uint8)
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
        scope_image = qimage.copy().mirrored(False, True)

        width = histo.shape[0]
        self.data_width = width

        centre_value, edge_positions = self._calculate_centre(histo)
        if centre_value is None or not np.isfinite(centre_value):
            centre_value = float(fit_gaussian(self.histo))
            edge_positions = None
        self.centre = float(centre_value)
        self.OnCentreChanged.emit(self.centre)

        if timing_active:
            timing_last = self._timing_checkpoint("centre", timing_last)

        a_sample = 0
        edge_lines: tuple[int, int] | None = None
        if width > 0 and self.analyser_widget_height > 0:
            a_sample = int(self.analyser_widget_height - self.centre * self.analyser_widget_height / width)
            if edge_positions is not None:
                edge_lines = tuple(
                    int(self.analyser_widget_height - edge * self.analyser_widget_height / width)
                    for edge in edge_positions
                )

        a_zero, a_text = 0, ""
        if getattr(self.parent_obj, "zero", 0.0) and width > 0:
            a_zero = int(self.analyser_widget_height - self.parent_obj.zero * self.analyser_widget_height / width)
            centre_real = (self.parent_obj.sensor_width / width) * (self.centre - self.parent_obj.zero)
            a_text = get_units(self.parent_obj.units, centre_real)

        frame_data = FrameData(scope_image, a_sample, a_zero, a_text, edge_lines)
        self.OnAnalyserUpdate.emit(frame_data)

        if timing_active:
            self._timing_checkpoint("histogram", timing_last)
            self._maybe_log_timing()

        self.ready = True

    def _to_qimage(self, gray: np.ndarray) -> QImage:
        h, w = gray.shape
        bytes_per_line = gray.strides[0]
        image = QImage(gray.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return image.copy()

    def _to_qimage_rgb(self, rgb: np.ndarray) -> QImage:
        h, w, _ = rgb.shape
        bytes_per_line = rgb.strides[0]
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return image.copy()

    def _calculate_centre(self, signal: np.ndarray) -> tuple[float | None, tuple[float, float] | None]:
        if self.measurement_mode == "Edge Midpoint":
            centre, edges = self._edge_midpoint(signal)
            if centre is not None:
                return centre, edges
        return float(fit_gaussian(signal)), None

    def _edge_midpoint(self, signal: np.ndarray) -> tuple[float | None, tuple[float, float] | None]:
        if signal.size < 3:
            return None, None
        max_val = float(np.nanmax(signal))
        min_val = float(np.nanmin(signal))
        if not np.isfinite(max_val) or not np.isfinite(min_val) or max_val <= min_val:
            return None, None
        normalized = (signal - min_val) / (max_val - min_val)
        threshold = 0.5
        binary = normalized > threshold
        transitions = np.diff(binary.astype(np.int8))
        rising = np.where(transitions == 1)[0]
        falling = np.where(transitions == -1)[0]

        left = None
        right = None

        if rising.size:
            left = self._interpolate_edge(normalized, rising[0], threshold)
        elif binary[0]:
            left = 0.0

        if falling.size:
            right = self._interpolate_edge(normalized, falling[-1], threshold)
        elif binary[-1]:
            right = float(len(normalized) - 1)

        if left is None or right is None or right <= left:
            return None, None

        centre = (left + right) / 2.0
        return centre, (left, right)

    def _timing_enabled(self) -> bool:
        if self._timing_base_enabled:
            return True
        if self.logging_until > 0.0:
            now = time.perf_counter()
            if now < self.logging_until:
                return True
            self.logging_until = 0.0
        return False

    def _timing_checkpoint(self, key: str, last_mark: float | None) -> float | None:
        now = time.perf_counter()
        if last_mark is not None:
            samples = self._timing_samples.get(key)
            if samples is not None:
                samples.append(now - last_mark)
        return now

    @Slot(float)
    def start_timed_logging(self, duration: float) -> None:
        if duration <= 0:
            return
        now = time.perf_counter()
        self.logging_until = max(self.logging_until, now + duration)
        self._timing_frames = 0
        for samples in self._timing_samples.values():
            samples.clear()
        self._timing_last_log = now

    def _maybe_log_timing(self) -> None:
        if not self._timing_enabled():
            return
        self._timing_frames += 1
        now = time.perf_counter()
        elapsed = now - self._timing_last_log
        if elapsed < 1.0:
            return
        averages = {
            key: (sum(samples) / len(samples) if samples else 0.0)
            for key, samples in self._timing_samples.items()
        }
        fps = self._timing_frames / elapsed if elapsed > 0 else 0.0
        msg = " ".join(f"{key}={averages[key] * 1000:.2f}ms" for key in self.TIMING_KEYS)
        log_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} fps={fps:.1f} {msg}\n"
        try:
            with self.log_path.open('a', encoding='utf-8') as fh:
                fh.write(log_line)
        except OSError:
            pass
        print(f"[FrameWorker] {log_line.strip()}", flush=True)
        for samples in self._timing_samples.values():
            samples.clear()
        self._timing_frames = 0
        self._timing_last_log = now
        if not self._timing_base_enabled and self.logging_until > 0.0 and now >= self.logging_until:
            self.logging_until = 0.0

    @staticmethod
    def _interpolate_edge(values: np.ndarray, index: int, threshold: float) -> float:
        if index < 0:
            return 0.0
        if index >= len(values) - 1:
            return float(len(values) - 1)
        v0 = float(values[index])
        v1 = float(values[index + 1])
        if v1 == v0:
            return float(index)
        return index + (threshold - v0) / (v1 - v0)


class FrameSender(QObject):  # type: ignore
    """
    A class to relay NumPy-based frames into the worker thread.
    """

    OnFrameChanged = Signal(object)
