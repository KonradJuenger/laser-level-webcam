# Phase 1: Critical Performance Optimizations (Week 1)

## Overview
Focus on the highest impact performance bottlenecks that are causing the most significant slowdowns in real-time processing.

## 1.1 Frame Processing Pipeline Optimization

**Current Issues (src/Workers.py, src/Core.py):**
- QImage to numpy conversions every frame
- Memory allocation for histogram arrays
- Expensive qimage2ndarray operations

**Optimizations:**
```python
# Pre-allocate buffers and reuse
class FrameWorker:
    def __init__(self):
        self.histogram_buffer = np.zeros(1080, dtype=np.float32)  # Pre-allocated
        self.frame_buffer = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.smoothing_kernel = None  # Cache smoothing kernel
        self.kernel_size = 0
        
    def process_frame_optimized(self, frame):
        # Direct buffer access instead of QImage conversion
        buffer_ptr = frame.bits()
        frame_array = np.frombuffer(buffer_ptr, dtype=np.uint8)
        
        # Reuse pre-allocated histogram buffer
        self.histogram_buffer.fill(0)
        
        # Process directly without intermediate conversions
        self._calculate_histogram_inplace(frame_array, self.histogram_buffer)
        
    def _cache_smoothing_kernel(self, size):
        if self.kernel_size != size:
            self.smoothing_kernel = self._generate_kernel(size)
            self.kernel_size = size
```

**Implementation Tasks:**
- [ ] Replace qimage2ndarray with direct buffer access
- [ ] Pre-allocate all processing arrays at startup
- [ ] Implement frame buffer pooling
- [ ] Cache smoothing kernels and reuse
- [ ] Add frame format validation and fast paths

**Expected Impact:** 40-60% reduction in frame processing time

## 1.2 Mathematical Algorithm Optimization

**Current Issues (src/curves.py, src/Core.py:22-63):**
- scipy.optimize.curve_fit is expensive for real-time use (10-50ms per call)
- Full linear regression recalculation for all samples
- Repeated statistical calculations

**Optimizations:**

### Fast Peak Detection
```python
import numba

@numba.jit(nopython=True, cache=True)
def fast_gaussian_center(histogram):
    """
    Ultra-fast weighted centroid calculation
    10x faster than curve fitting for laser line detection
    """
    total_intensity = 0.0
    weighted_sum = 0.0
    
    for i in range(len(histogram)):
        intensity = histogram[i]
        if intensity > 0:
            total_intensity += intensity
            weighted_sum += i * intensity
            
    if total_intensity > 0:
        return weighted_sum / total_intensity
    return len(histogram) // 2

@numba.jit(nopython=True, cache=True)
def fast_peak_detection_with_noise_filter(histogram, noise_threshold=0.1):
    """
    Enhanced peak detection with noise filtering
    Optimized for laser line characteristics
    """
    max_val = np.max(histogram)
    threshold = max_val * noise_threshold
    
    # Filter noise
    filtered = np.where(histogram > threshold, histogram, 0.0)
    
    return fast_gaussian_center(filtered)
```

### Incremental Statistics
```python
class IncrementalLinearRegression:
    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_xy = 0.0
        self.sum_yy = 0.0
        
    def add_sample(self, x, y):
        """O(1) sample addition instead of O(n) recalculation"""
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_xy += x * y
        self.sum_yy += y * y
        
    def remove_sample(self, x, y):
        """O(1) sample removal for sliding window"""
        if self.n > 0:
            self.n -= 1
            self.sum_x -= x
            self.sum_y -= y
            self.sum_xx -= x * x
            self.sum_xy -= x * y
            self.sum_yy -= y * y
            
    def get_slope_intercept(self):
        """Calculate slope and intercept in O(1)"""
        if self.n < 2:
            return 0.0, 0.0
            
        denominator = self.n * self.sum_xx - self.sum_x * self.sum_x
        if abs(denominator) < 1e-10:
            return 0.0, self.sum_y / self.n
            
        slope = (self.n * self.sum_xy - self.sum_x * self.sum_y) / denominator
        intercept = (self.sum_y - slope * self.sum_x) / self.n
        
        return slope, intercept
```

**Implementation Tasks:**
- [ ] Replace scipy curve_fit with fast weighted centroid
- [ ] Implement incremental linear regression class
- [ ] Add numba JIT compilation for mathematical functions
- [ ] Pre-calculate lookup tables for common operations
- [ ] Add noise filtering optimized for laser characteristics

**Expected Impact:** 80-90% reduction in mathematical calculation time

## 1.3 Memory Management Optimization

**Current Issues:**
- Continuous Sample object allocation (creating garbage collection pressure)
- String formatting overhead in UI updates
- Numpy array recreation every frame

**Optimizations:**

### Object Pooling
```python
class SamplePool:
    def __init__(self, size=1000):
        self.pool = [Sample(0, 0.0) for _ in range(size)]
        self.available = list(range(size))
        self.in_use = set()
        
    def get_sample(self):
        if self.available:
            idx = self.available.pop()
            sample = self.pool[idx]
            self.in_use.add(idx)
            return sample, idx
        # Fallback - should rarely happen
        return Sample(0, 0.0), -1
        
    def return_sample(self, idx):
        if idx >= 0 and idx in self.in_use:
            sample = self.pool[idx]
            # Reset sample data
            sample.x = 0
            sample.y = 0.0
            sample.linYError = 0.0
            sample.shim = 0.0
            sample.scrape = 0.0
            
            self.in_use.remove(idx)
            self.available.append(idx)

class CircularBuffer:
    def __init__(self, size, dtype=np.float32):
        self.buffer = np.zeros(size, dtype=dtype)
        self.size = size
        self.index = 0
        self.filled = False
        
    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.filled = True
            
    def get_data(self):
        if not self.filled:
            return self.buffer[:self.index]
        # Return data in correct order
        return np.concatenate([
            self.buffer[self.index:],
            self.buffer[:self.index]
        ])
```

### String Caching
```python
class StringCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        
    def format_measurement(self, value, units):
        # Cache formatted strings to avoid repeated formatting
        key = (round(value, 6), units)
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                # Simple LRU: clear oldest half
                keys_to_remove = list(self.cache.keys())[:self.max_size // 2]
                for k in keys_to_remove:
                    del self.cache[k]
                    
            self.cache[key] = f"{value:.6f} {units}"
        return self.cache[key]
```

**Implementation Tasks:**
- [ ] Implement Sample object pooling
- [ ] Create circular buffers for histogram data
- [ ] Add string caching for UI updates
- [ ] Use memory-mapped arrays for large datasets
- [ ] Implement buffer reuse strategies

**Expected Impact:** 30-50% reduction in memory allocations and GC pressure

## Implementation Timeline - Week 1

### Days 1-2: Frame Processing Pipeline
- Replace QImage conversions with direct buffer access
- Implement buffer pre-allocation and reuse
- Add smoothing kernel caching

### Days 3-4: Mathematical Optimization
- Replace curve fitting with fast peak detection
- Implement incremental linear regression
- Add numba JIT compilation

### Days 5-7: Memory Management
- Implement object pooling system
- Add circular buffers for data storage
- Create string caching system
- Integration testing and performance validation

## Testing and Validation

### Performance Benchmarks
```python
import time

class PerformanceBenchmark:
    def __init__(self):
        self.timers = {}
        
    def start_timer(self, name):
        self.timers[name] = time.perf_counter()
        
    def end_timer(self, name):
        if name in self.timers:
            elapsed = time.perf_counter() - self.timers[name]
            print(f"{name}: {elapsed*1000:.2f}ms")
            return elapsed
        return 0
```

### Success Criteria
- Frame processing time: <16ms (target: 10ms)
- Mathematical calculations: <5ms (target: 2ms)
- Memory allocations: <50% of original
- Overall frame rate: 20+ FPS by end of Phase 1

## Risk Mitigation
- Keep original algorithms as fallback options
- Comprehensive unit testing for mathematical accuracy
- Memory leak detection and monitoring
- Performance regression testing