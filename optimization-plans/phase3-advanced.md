# Phase 3: Advanced Optimizations (Week 3)

## Overview
Prepare the codebase for Nuitka compilation and implement advanced optimizations for maximum performance.

## 3.1 Nuitka Compilation Optimization

**Current Issues:**
- Dynamic attribute access slowing compilation
- Missing type hints reducing optimization potential
- Import overhead in critical paths
- Suboptimal data structures for compiled code

**Optimizations:**

### Compilation Settings
```bash
#!/bin/bash
# optimal-build.sh - Optimized Nuitka compilation script

nuitka --standalone \
       --enable-plugin=pyside6 \
       --enable-plugin=numpy \
       --assume-yes-for-downloads \
       --output-dir=dist \
       --jobs=8 \
       --lto=yes \
       --python-flag=O \
       --python-flag=OO \
       --remove-output \
       --follow-imports \
       --include-package=numpy \
       --include-package=scipy \
       --include-package=matplotlib \
       --onefile-tempdir-spec="%TEMP%/laser_level_app" \
       --windows-icon-from-ico=src/icon/laser-beam.ico \
       --product-name="LaserVision" \
       --file-version="2.0.0" \
       --company-name="LaserLevel Tools" \
       --file-description="High-precision laser measurement tool" \
       laser-level-webcam.py
```

### Type Annotations for Better Optimization
```python
from __future__ import annotations
from typing import Optional, List, Tuple, Union, Protocol
import numpy as np
import numpy.typing as npt

# Type aliases for better performance
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]
HistogramData = npt.NDArray[np.float32]

class FrameProcessor(Protocol):
    def process_frame(self, frame: bytes) -> Tuple[float, FloatArray]: ...

class OptimizedCore:
    def __init__(self) -> None:
        self.samples: List[Sample] = []
        self.zero: float = 0.0
        self.sensor_width: float = 5.9
        self.units: str = "micrometers"
        
        # Pre-allocate arrays with specific types
        self.histogram_buffer: FloatArray = np.zeros(1080, dtype=np.float32)
        self.smoothing_kernel: Optional[FloatArray] = None
        
    def process_measurement(self, center_pixel: float) -> float:
        """Type-annotated measurement processing"""
        size_in_mm: float = (self.sensor_width / 1080.0) * (center_pixel - self.zero)
        return size_in_mm

# Use __slots__ for better memory efficiency and faster attribute access
class Sample:
    __slots__ = ('x', 'y', 'linYError', 'shim', 'scrape')
    
    def __init__(self, x: int, y: float) -> None:
        self.x: int = x
        self.y: float = y
        self.linYError: float = 0.0
        self.shim: float = 0.0
        self.scrape: float = 0.0
```

### Import Optimization
```python
# optimized_imports.py - Centralized import management
"""
Optimized import module to reduce import overhead in critical paths
"""

# Pre-import commonly used modules
import numpy as np
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Constants for compilation optimization
HISTOGRAM_SIZE: int = 1080
MAX_SAMPLES: int = 1000
UPDATE_INTERVAL_MS: int = 33  # ~30 FPS

# Pre-compiled regular expressions if needed
import re
FLOAT_PATTERN = re.compile(r'^-?\d+\.?\d*$')

# Function type definitions for better optimization
ProcessingFunction = callable[[np.ndarray], Tuple[float, np.ndarray]]
UpdateFunction = callable[[float], None]
```

**Implementation Tasks:**
- [ ] Add comprehensive type hints throughout codebase
- [ ] Implement __slots__ for performance-critical classes
- [ ] Optimize import structure and reduce dynamic imports
- [ ] Create compilation script with optimal flags
- [ ] Profile compiled vs interpreted performance

**Expected Impact:** 20-40% performance improvement after compilation

## 3.2 Algorithm-Specific Optimizations

**Current Issues:**
- Generic algorithms not optimized for laser line characteristics
- Unnecessary precision in calculations
- No hardware-specific optimizations

**Optimizations:**

### Laser Line Optimized Peak Detection
```python
import numba
from numba import types

@numba.jit(nopython=True, cache=True, fastmath=True)
def laser_line_center_optimized(histogram: np.ndarray, 
                              noise_threshold: float = 0.05,
                              smoothing_passes: int = 1) -> Tuple[float, float]:
    """
    Ultra-optimized laser line center detection
    Specifically tuned for laser line characteristics:
    - Sharp peak with gaussian-like shape
    - Low noise baseline
    - Consistent intensity patterns
    """
    length = len(histogram)
    
    # Fast noise threshold calculation
    max_val = 0.0
    for i in range(length):
        if histogram[i] > max_val:
            max_val = histogram[i]
    
    threshold = max_val * noise_threshold
    
    # Apply in-place smoothing if needed
    if smoothing_passes > 0:
        for _ in range(smoothing_passes):
            for i in range(1, length - 1):
                histogram[i] = (histogram[i-1] + histogram[i] + histogram[i+1]) / 3.0
    
    # Find region of interest (ROI) around peak
    peak_idx = 0
    for i in range(length):
        if histogram[i] > histogram[peak_idx]:
            peak_idx = i
    
    # Define ROI around peak (adaptive width)
    roi_half_width = max(10, int(max_val * 0.1))  # Adaptive ROI
    roi_start = max(0, peak_idx - roi_half_width)
    roi_end = min(length, peak_idx + roi_half_width)
    
    # Weighted centroid calculation within ROI
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i in range(roi_start, roi_end):
        intensity = histogram[i]
        if intensity > threshold:
            weight = intensity * intensity  # Square weighting for sharper peak
            total_weight += weight
            weighted_sum += i * weight
    
    if total_weight > 0:
        center = weighted_sum / total_weight
        confidence = total_weight / (roi_end - roi_start)  # Confidence metric
        return center, confidence
    
    return float(peak_idx), 0.0

@numba.jit(nopython=True, cache=True)
def fast_histogram_calculation(frame_data: np.ndarray, 
                             width: int, 
                             height: int,
                             histogram: np.ndarray) -> None:
    """
    Optimized histogram calculation for grayscale conversion and row averaging
    """
    # Clear histogram
    for i in range(len(histogram)):
        histogram[i] = 0.0
    
    # Process frame data (assuming BGR format)
    for y in range(height):
        for x in range(width):
            # Fast grayscale conversion (weighted average)
            pixel_idx = (y * width + x) * 3
            b = frame_data[pixel_idx]
            g = frame_data[pixel_idx + 1]
            r = frame_data[pixel_idx + 2]
            
            # Fast grayscale: 0.299*R + 0.587*G + 0.114*B
            # Approximated with integer math for speed
            gray = (r * 299 + g * 587 + b * 114) // 1000
            
            histogram[y] += gray
    
    # Average across width
    for y in range(height):
        histogram[y] /= width
```

### Real-time Statistics with Fixed-Point Math
```python
@numba.jit(nopython=True, cache=True)
def incremental_regression_update(n: int,
                                sum_x: float, sum_y: float,
                                sum_xx: float, sum_xy: float,
                                new_x: float, new_y: float) -> Tuple[int, float, float, float, float, float, float]:
    """
    Ultra-fast incremental linear regression update
    Returns: (new_n, new_sum_x, new_sum_y, new_sum_xx, new_sum_xy, slope, intercept)
    """
    n += 1
    sum_x += new_x
    sum_y += new_y
    sum_xx += new_x * new_x
    sum_xy += new_x * new_y
    
    # Calculate slope and intercept
    if n < 2:
        return n, sum_x, sum_y, sum_xx, sum_xy, 0.0, 0.0
    
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-10:
        slope = 0.0
        intercept = sum_y / n
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
    
    return n, sum_x, sum_y, sum_xx, sum_xy, slope, intercept

class FastSlidingWindow:
    """Optimized sliding window for real-time statistics"""
    __slots__ = ('data', 'size', 'index', 'filled', 'sum_val', 'sum_sq')
    
    def __init__(self, size: int) -> None:
        self.data = np.zeros(size, dtype=np.float32)
        self.size = size
        self.index = 0
        self.filled = False
        self.sum_val = 0.0
        self.sum_sq = 0.0
    
    @numba.jit(nopython=True)
    def add_value(self, value: float) -> Tuple[float, float]:
        """Add value and return mean, std in O(1)"""
        old_value = self.data[self.index]
        
        # Update sums
        if self.filled:
            self.sum_val += value - old_value
            self.sum_sq += value * value - old_value * old_value
        else:
            self.sum_val += value
            self.sum_sq += value * value
        
        # Store new value
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size
        
        if self.index == 0:
            self.filled = True
        
        # Calculate statistics
        count = self.size if self.filled else self.index
        if count == 0:
            return 0.0, 0.0
            
        mean = self.sum_val / count
        variance = (self.sum_sq / count) - (mean * mean)
        std = np.sqrt(max(0.0, variance))
        
        return mean, std
```

**Implementation Tasks:**
- [ ] Implement laser-specific peak detection algorithm
- [ ] Add numba compilation for all mathematical functions
- [ ] Create fixed-point arithmetic where appropriate
- [ ] Optimize for specific camera sensor characteristics
- [ ] Add lookup tables for common calculations

**Expected Impact:** 60-80% improvement in mathematical calculation speed

## 3.3 Hardware-Specific Optimizations

**Current Issues:**
- Generic camera interface not optimized for specific hardware
- No SIMD utilization for array operations
- Cache-inefficient memory access patterns

**Optimizations:**

### Camera-Specific Fast Paths
```python
class CameraOptimizer:
    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.fast_path_available = False
        self.buffer_format = None
        
        # Detect camera-specific optimizations
        self._detect_optimizations()
    
    def _detect_optimizations(self):
        """Detect available camera-specific optimizations"""
        known_optimizations = {
            'Logitech': self._setup_logitech_optimization,
            'Microsoft': self._setup_microsoft_optimization,
            'USB2.0': self._setup_usb_optimization,
        }
        
        for camera_type, setup_func in known_optimizations.items():
            if camera_type.lower() in self.camera_name.lower():
                setup_func()
                break
    
    def _setup_logitech_optimization(self):
        """Logitech camera specific optimizations"""
        self.fast_path_available = True
        self.buffer_format = 'YUYV'  # Many Logitech cameras use YUYV
        
    def get_optimized_processor(self):
        """Return camera-specific frame processor"""
        if self.buffer_format == 'YUYV':
            return self._process_yuyv_frame
        return self._process_generic_frame
    
    @numba.jit(nopython=True, cache=True)
    def _process_yuyv_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """Optimized YUYV processing"""
        # YUYV format: Y0 U Y1 V (4 bytes for 2 pixels)
        # We only need Y (luminance) for our histogram
        height = 1080  # Assuming known resolution
        histogram = np.zeros(height, dtype=np.float32)
        
        # Extract Y values efficiently
        for y in range(height):
            row_sum = 0.0
            for x in range(0, 1920 * 2, 4):  # YUYV is 2 bytes per pixel
                y_idx = y * 1920 * 2 + x
                if y_idx < len(frame_data):
                    y0 = frame_data[y_idx]      # First Y
                    y1 = frame_data[y_idx + 2]  # Second Y
                    row_sum += y0 + y1
            
            histogram[y] = row_sum / 1920.0
        
        return histogram
```

### SIMD-Optimized Operations
```python
# Use numpy's built-in SIMD optimizations more effectively
@numba.jit(nopython=True, cache=True, parallel=True)
def parallel_histogram_processing(frame_data: np.ndarray, 
                                width: int, 
                                height: int) -> np.ndarray:
    """
    Parallel processing of frame data using numba's parallel features
    """
    histogram = np.zeros(height, dtype=np.float32)
    
    # Parallel processing of rows
    for y in numba.prange(height):
        row_sum = 0.0
        row_start = y * width * 3
        
        # Vectorized sum of row pixels
        for x in range(width):
            pixel_idx = row_start + x * 3
            # Fast grayscale conversion
            gray = (frame_data[pixel_idx] * 299 + 
                   frame_data[pixel_idx + 1] * 587 + 
                   frame_data[pixel_idx + 2] * 114) // 1000
            row_sum += gray
        
        histogram[y] = row_sum / width
    
    return histogram

# Memory-aligned data structures for cache efficiency
class AlignedBuffer:
    def __init__(self, size: int, dtype=np.float32):
        # Align to 64-byte boundaries for optimal cache performance
        self.buffer = np.empty(size + 16, dtype=dtype)
        offset = self.buffer.ctypes.data % 64
        if offset != 0:
            offset = 64 - offset
        self.aligned_view = self.buffer[offset//dtype().itemsize:offset//dtype().itemsize + size]
    
    def get_buffer(self) -> np.ndarray:
        return self.aligned_view
```

### Memory Layout Optimization
```python
class OptimizedDataStructures:
    """Memory-efficient data structures optimized for performance"""
    
    def __init__(self, max_samples: int = 1000):
        # Structure of Arrays (SoA) instead of Array of Structures (AoS)
        # Better for SIMD operations and cache efficiency
        self.sample_x = np.zeros(max_samples, dtype=np.int32)
        self.sample_y = np.zeros(max_samples, dtype=np.float32)
        self.sample_errors = np.zeros(max_samples, dtype=np.float32)
        self.sample_shims = np.zeros(max_samples, dtype=np.float32)
        self.sample_scrapes = np.zeros(max_samples, dtype=np.float32)
        
        self.sample_count = 0
        self.max_samples = max_samples
        
        # Pre-allocated working arrays
        self.work_array_1 = np.zeros(max_samples, dtype=np.float32)
        self.work_array_2 = np.zeros(max_samples, dtype=np.float32)
    
    @numba.jit(nopython=True)
    def add_sample(self, x: int, y: float) -> bool:
        """Add sample using optimized data layout"""
        if self.sample_count >= self.max_samples:
            return False
            
        idx = self.sample_count
        self.sample_x[idx] = x
        self.sample_y[idx] = y
        self.sample_count += 1
        return True
    
    @numba.jit(nopython=True)
    def calculate_regression_vectorized(self) -> Tuple[float, float]:
        """Vectorized linear regression calculation"""
        if self.sample_count < 2:
            return 0.0, 0.0
        
        n = self.sample_count
        
        # Use pre-allocated work arrays for intermediate calculations
        x_data = self.sample_x[:n].astype(np.float32)
        y_data = self.sample_y[:n]
        
        # Vectorized calculations
        sum_x = np.sum(x_data)
        sum_y = np.sum(y_data)
        sum_xx = np.sum(x_data * x_data)
        sum_xy = np.sum(x_data * y_data)
        
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0, sum_y / n
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
```

**Implementation Tasks:**
- [ ] Implement camera-specific optimizations
- [ ] Add SIMD-optimized array operations
- [ ] Optimize memory layout for cache efficiency
- [ ] Create hardware detection and optimization selection
- [ ] Add performance profiling for different hardware configurations

**Expected Impact:** 30-50% improvement on specific hardware configurations

## Implementation Timeline - Week 3

### Days 1-3: Nuitka Optimization
- Add comprehensive type hints
- Implement __slots__ for critical classes
- Create optimized compilation script
- Test compilation performance

### Days 4-5: Algorithm Optimization
- Implement laser-specific algorithms
- Add numba compilation for math functions
- Create vectorized operations
- Benchmark algorithm improvements

### Days 6-7: Hardware Optimization
- Implement camera-specific optimizations
- Add SIMD operations where beneficial
- Optimize memory layouts
- Integration testing and validation

## Testing and Validation

### Performance Benchmarking
```python
class CompilationBenchmark:
    def __init__(self):
        self.interpreted_times = []
        self.compiled_times = []
        
    def benchmark_function(self, func, args, iterations=1000):
        """Benchmark function performance"""
        # Interpreted
        start = time.perf_counter()
        for _ in range(iterations):
            func(*args)
        interpreted_time = time.perf_counter() - start
        
        return interpreted_time / iterations

    def compare_performance(self, old_func, new_func, test_data):
        """Compare old vs new implementation"""
        old_time = self.benchmark_function(old_func, test_data)
        new_time = self.benchmark_function(new_func, test_data)
        
        improvement = (old_time - new_time) / old_time * 100
        print(f"Performance improvement: {improvement:.1f}%")
        
        return improvement
```

### Success Criteria
- Compilation successful with all optimizations
- 20%+ performance improvement after compilation
- Algorithm accuracy maintained within 0.1%
- Memory usage stable or improved
- Hardware-specific optimizations provide measurable benefits

## Risk Mitigation
- Maintain interpreted mode as fallback
- Extensive accuracy testing for optimized algorithms
- Hardware compatibility testing
- Performance regression detection