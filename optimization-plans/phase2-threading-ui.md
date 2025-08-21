# Phase 2: Threading and UI Optimization (Week 2)

## Overview
Optimize the threading architecture and UI updates to achieve smooth real-time performance with minimal latency.

## 2.1 Threading Architecture Redesign

**Current Issues (src/Workers.py):**
- Qt signal/slot overhead causing latency
- Thread synchronization bottlenecks
- Processing backlog causing frame drops
- No intelligent frame management

**Optimizations:**

### Lock-Free Queue System
```python
import queue
import threading
import time
from collections import deque
import numpy as np

class FrameProcessor:
    def __init__(self, max_queue_size=3):
        # Drop old frames if processing falls behind
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=10)
        self.stats_queue = queue.Queue(maxsize=100)
        
        self.processing_thread = None
        self.running = False
        
        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
    def start(self):
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
    def submit_frame(self, frame, timestamp):
        """Submit frame for processing, drop if queue full"""
        try:
            # Drop oldest frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put_nowait((frame, timestamp))
            return True
        except queue.Full:
            return False
            
    def get_result(self):
        """Non-blocking result retrieval"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
            
    def _process_loop(self):
        """Main processing loop - runs in separate thread"""
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=0.1)
                
                start_time = time.perf_counter()
                result = self._fast_process_frame(frame)
                processing_time = time.perf_counter() - start_time
                
                # Store results with timing
                self.result_queue.put({
                    'result': result,
                    'timestamp': timestamp,
                    'processing_time': processing_time
                })
                
                # Update performance stats
                self.processing_times.append(processing_time * 1000)  # ms
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                
    def get_performance_stats(self):
        """Get current performance metrics"""
        if not self.processing_times:
            return {'avg_ms': 0, 'fps': 0, 'queue_size': 0}
            
        avg_time_ms = np.mean(self.processing_times)
        fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
        
        return {
            'avg_processing_ms': avg_time_ms,
            'estimated_fps': fps,
            'queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
```

### Intelligent Frame Dropping
```python
class AdaptiveFrameManager:
    def __init__(self):
        self.target_fps = 30
        self.current_fps = 0
        self.frame_drop_ratio = 0
        self.performance_history = deque(maxlen=30)
        
    def should_process_frame(self, current_time):
        """Decide whether to process this frame based on performance"""
        # Update FPS calculation
        self.performance_history.append(current_time)
        
        if len(self.performance_history) >= 2:
            time_diff = self.performance_history[-1] - self.performance_history[0]
            self.current_fps = len(self.performance_history) / time_diff
            
        # Adaptive frame dropping
        if self.current_fps < self.target_fps * 0.8:
            # Performance is poor, increase frame dropping
            self.frame_drop_ratio = min(0.5, self.frame_drop_ratio + 0.1)
        elif self.current_fps > self.target_fps * 0.95:
            # Performance is good, reduce frame dropping
            self.frame_drop_ratio = max(0, self.frame_drop_ratio - 0.05)
            
        # Random dropping based on ratio
        import random
        return random.random() > self.frame_drop_ratio
```

**Implementation Tasks:**
- [ ] Replace Qt signals with lock-free queues
- [ ] Implement intelligent frame dropping system
- [ ] Add performance monitoring and adaptive behavior
- [ ] Create separate threads for acquisition, processing, and display
- [ ] Add frame timing and latency measurement

**Expected Impact:** 50-70% reduction in threading overhead

## 2.2 Graph Update Optimization

**Current Issues (src/Widgets.py):**
- Full matplotlib redraw every update (expensive)
- No caching of static elements
- Updates triggered too frequently
- No dirty flag system

**Optimizations:**

### Incremental Graph Updates with Blitting
```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class OptimizedGraph:
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = self.fig.canvas
        
        # Cache for static elements
        self.cached_background = None
        self.static_elements_dirty = True
        
        # Data elements that change frequently
        self.data_line = None
        self.peak_marker = None
        self.current_data = np.array([])
        
        # Dirty flags for selective updates
        self.dirty_flags = {
            'background': True,
            'data': True,
            'peak': True,
            'annotations': True
        }
        
        # Update throttling
        self.last_update_time = 0
        self.min_update_interval = 1.0 / 15.0  # 15 Hz max updates
        
        self._setup_static_elements()
        
    def _setup_static_elements(self):
        """Setup static graph elements that don't change often"""
        self.ax.set_xlim(0, 1080)
        self.ax.set_ylim(0, 255)
        self.ax.set_xlabel('Pixel Position')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(True, alpha=0.3)
        
        # Create data line (will be updated frequently)
        self.data_line, = self.ax.plot([], [], 'b-', linewidth=1.5)
        self.peak_marker, = self.ax.plot([], [], 'ro', markersize=8)
        
        self.fig.tight_layout()
        
    def _cache_background(self):
        """Cache the static background for fast blitting"""
        self.canvas.draw()
        self.cached_background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.static_elements_dirty = False
        
    def update_data(self, histogram, peak_position=None):
        """Update only the data, using cached background"""
        current_time = time.time()
        
        # Throttle updates to prevent overwhelming the UI
        if current_time - self.last_update_time < self.min_update_interval:
            return
            
        self.last_update_time = current_time
        
        # Cache background if needed
        if self.static_elements_dirty or self.cached_background is None:
            self._cache_background()
            
        # Restore cached background
        self.canvas.restore_region(self.cached_background)
        
        # Update data line
        x_data = np.arange(len(histogram))
        self.data_line.set_data(x_data, histogram)
        self.ax.draw_artist(self.data_line)
        
        # Update peak marker if provided
        if peak_position is not None:
            peak_y = histogram[int(peak_position)] if int(peak_position) < len(histogram) else 0
            self.peak_marker.set_data([peak_position], [peak_y])
            self.ax.draw_artist(self.peak_marker)
            
        # Blit only the changed area
        self.canvas.blit(self.ax.bbox)
        
    def force_full_redraw(self):
        """Force a complete redraw (use sparingly)"""
        self.static_elements_dirty = True
        self.canvas.draw()
```

### Batched UI Updates
```python
class UIUpdateBatcher:
    def __init__(self, update_interval=1.0/10.0):  # 10 Hz
        self.update_interval = update_interval
        self.pending_updates = {}
        self.timer = None
        self.last_update = 0
        
    def schedule_update(self, component, data):
        """Schedule an update, batching multiple requests"""
        self.pending_updates[component] = data
        
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self._execute_updates()
            
    def _execute_updates(self):
        """Execute all pending updates in batch"""
        if not self.pending_updates:
            return
            
        # Process all updates at once
        for component, data in self.pending_updates.items():
            try:
                component.update_display(data)
            except Exception as e:
                print(f"Update error for {component}: {e}")
                
        self.pending_updates.clear()
        self.last_update = time.time()
```

**Implementation Tasks:**
- [ ] Implement matplotlib blitting for incremental updates
- [ ] Cache static graph elements (axes, labels, grid)
- [ ] Add dirty flag system for selective rendering
- [ ] Implement update throttling and batching
- [ ] Create separate update queues for different UI components

**Expected Impact:** 70-80% reduction in graph update time

## 2.3 UI Thread Optimization

**Current Issues (src/main.py):**
- Heavy calculations on main UI thread
- Table widget updates every frame
- Excessive signal emissions causing lag
- No update prioritization

**Optimizations:**

### UI Thread Separation
```python
class UIController:
    def __init__(self, main_window):
        self.main_window = main_window
        self.update_queue = queue.Queue()
        self.update_timer = QTimer()
        
        # Different update frequencies for different components
        self.update_frequencies = {
            'graph': 15,        # 15 Hz
            'table': 5,         # 5 Hz
            'status': 2,        # 2 Hz
            'settings': 1       # 1 Hz
        }
        
        self.last_updates = {component: 0 for component in self.update_frequencies}
        
        # Setup update timer
        self.update_timer.timeout.connect(self._process_ui_updates)
        self.update_timer.start(1000 // 60)  # 60 Hz timer, but components update at their own rates
        
    def schedule_update(self, component, data, priority=1):
        """Schedule UI update with priority"""
        current_time = time.time()
        component_freq = self.update_frequencies.get(component, 10)
        
        # Check if enough time has passed for this component
        if current_time - self.last_updates[component] >= 1.0 / component_freq:
            self.update_queue.put((priority, component, data, current_time))
            
    def _process_ui_updates(self):
        """Process queued UI updates in priority order"""
        updates_processed = 0
        max_updates_per_cycle = 5  # Prevent UI thread blocking
        
        try:
            while updates_processed < max_updates_per_cycle:
                priority, component, data, timestamp = self.update_queue.get_nowait()
                
                # Execute the update
                self._execute_component_update(component, data)
                self.last_updates[component] = timestamp
                updates_processed += 1
                
        except queue.Empty:
            pass
            
    def _execute_component_update(self, component, data):
        """Execute specific component update"""
        if component == 'graph':
            self.main_window.graph.update_data(data.get('histogram'), data.get('peak'))
        elif component == 'table':
            self._update_table_efficiently(data)
        elif component == 'status':
            self.main_window.status_bar.showMessage(data['message'])
        # Add more component handlers as needed
        
    def _update_table_efficiently(self, data):
        """Efficient table updates with minimal redraws"""
        table = self.main_window.sample_table
        
        # Block signals during batch update
        table.blockSignals(True)
        
        try:
            # Update only changed cells
            for row, row_data in enumerate(data.get('rows', [])):
                if row >= table.rowCount():
                    table.insertRow(row)
                    
                for col, value in enumerate(row_data):
                    current_item = table.item(row, col)
                    if current_item is None or current_item.text() != str(value):
                        if current_item is None:
                            current_item = QTableWidgetItem()
                            table.setItem(row, col, current_item)
                        current_item.setText(str(value))
                        
        finally:
            table.blockSignals(False)
```

### Efficient Settings Management
```python
class SettingsManager:
    def __init__(self):
        self.settings = QSettings("laser-level-webcam", "LaserLevelWebcam")
        self.cached_values = {}
        self.dirty_settings = set()
        
        # Batch save timer
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self._flush_settings)
        self.save_timer.setSingleShot(True)
        
    def set_value(self, key, value):
        """Set value with delayed writing"""
        self.cached_values[key] = value
        self.dirty_settings.add(key)
        
        # Restart timer for batched saving
        self.save_timer.start(1000)  # Save after 1 second of no changes
        
    def get_value(self, key, default=None):
        """Get value from cache or settings"""
        if key in self.cached_values:
            return self.cached_values[key]
            
        value = self.settings.value(key, default)
        self.cached_values[key] = value
        return value
        
    def _flush_settings(self):
        """Write all dirty settings to disk"""
        for key in self.dirty_settings:
            if key in self.cached_values:
                self.settings.setValue(key, self.cached_values[key])
                
        self.dirty_settings.clear()
        self.settings.sync()
```

**Implementation Tasks:**
- [ ] Separate heavy calculations from UI thread
- [ ] Implement component-specific update frequencies
- [ ] Add update prioritization and batching
- [ ] Create efficient table update mechanisms
- [ ] Implement delayed settings saving

**Expected Impact:** 40-60% reduction in UI thread blocking

## Implementation Timeline - Week 2

### Days 1-3: Threading Architecture
- Implement lock-free queue system
- Add intelligent frame dropping
- Create performance monitoring
- Replace Qt signal/slot with direct communication

### Days 4-5: Graph Optimization
- Implement matplotlib blitting
- Add background caching
- Create update throttling system
- Optimize graph data structures

### Days 6-7: UI Thread Optimization
- Separate UI updates from processing
- Implement batched updates
- Add component-specific update rates
- Integration testing and performance validation

## Testing and Validation

### Performance Metrics
```python
class ThreadingBenchmark:
    def __init__(self):
        self.frame_latencies = deque(maxlen=100)
        self.ui_update_times = deque(maxlen=100)
        self.queue_sizes = deque(maxlen=100)
        
    def measure_frame_latency(self, frame_timestamp, result_timestamp):
        latency = result_timestamp - frame_timestamp
        self.frame_latencies.append(latency * 1000)  # ms
        
    def get_performance_report(self):
        return {
            'avg_latency_ms': np.mean(self.frame_latencies) if self.frame_latencies else 0,
            'avg_ui_update_ms': np.mean(self.ui_update_times) if self.ui_update_times else 0,
            'avg_queue_size': np.mean(self.queue_sizes) if self.queue_sizes else 0
        }
```

### Success Criteria
- End-to-end latency: <50ms
- Graph updates: 15 Hz sustained
- UI responsiveness: No blocking >16ms
- Queue management: <3 frames backlog average

## Risk Mitigation
- Maintain Qt signal/slot as fallback option
- Comprehensive threading safety testing
- UI responsiveness monitoring
- Memory usage validation with new threading model