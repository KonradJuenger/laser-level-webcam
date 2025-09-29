# Agent Notes

## Mission
- Capture a quick briefing on the laser-level-webcam project for future agents.
- Summarize the moving parts, external dependencies, and entry points.

## High-Level Overview
- The project turns a USB webcam into a laser measurement tool with a PySide6 GUI.
- It reads video frames, extracts a 1D brightness profile, fits a Gaussian curve, and converts pixel offsets into physical units.
- Collected samples are filtered, averaged, stored, and visualized for surface flatness analysis.
- A companion GUI (linuxcnc_remote_driver.py) can drive CNC probing workflows and plot captured height maps.

## Key Modules
- src/main.py: Main GUI with camera controls, sampling workflow, table/graph views, CSV export, socket control, and cyclic sampling.
- src/Core.py: Handles camera capture threads, histogram processing, Gaussian fitting results, unit conversion, and sample management.
- src/Workers.py: Background workers for frame-to-histogram processing and statistical filtering of subsamples.
- src/Widgets.py: Custom Qt/Matplotlib widgets for live video, analyser overlay, and plotted samples.
- src/linuxcnc_remote_driver.py: Plotly + Qt GUI for remote LinuxCNC probing jobs using classes under src/CNC_jobs.

## External Dependencies
- PySide6 for GUI, multimedia capture, threads, and networking.
- NumPy, SciPy, qimage2ndarray for numerical work; Matplotlib and Plotly for visualization.
- Optional ffmpeg for hardware camera settings dialog; sockets for remote commands.

## Operational Notes
- Measurements require zeroing before sampling; subsample counts and outlier removal are configurable.
- TCP socket server (see src/s_server.py) exposes ZERO and TAKE_SAMPLE commands for automation.
- Settings persist via QSettings, so mind side effects when testing.
- Plotly graphs in the CNC driver can be saved/loaded/exported; data stored as NumPy arrays or CSV.

## Open Questions
- LinuxCNC integration stubs (ProbeJob etc.) warrant deeper review if CNC automation is in scope.
- Unit map in src/utils.py includes a garbled key ("I?m"); confirm intended symbol before changing conversions.

## Benchmarking
- Run `python benchmarks/baseline.py` to record baseline timings for Gaussian fitting and sample regression paths. Adjust CLI flags to explore different frame widths or sample counts.

