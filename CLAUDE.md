# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LaserVision is a Python application that creates a highly accurate measurement device using a webcam sensor and laser level. The tool can achieve measurements between 0.5-2 micrometers by analyzing laser beam intensity values on a webcam sensor with its lens removed.

## Development Commands

### Installation and Setup
```bash
# Create virtual environment
virtualenv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies (now includes development tools)
pip install -r requirements.txt
```

### Running the Application
```bash
# Main webcam sensor application
python laser-level-webcam.py
# or
python src/main.py

# Windows batch launcher
launch.bat

# Via pip installation
laser-level-webcam
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/curves_test.py
pytest tests/main_test.py
pytest tests/utils_test.py
pytest tests/widgets_test.py

# Run with coverage
coverage run -m pytest tests/
coverage report
```

### Code Quality
```bash
# Run pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks (one-time setup)
pre-commit install
```

## Architecture Overview

### Main Components

**Core Architecture:**
- `src/main.py`: Main GUI application entry point using PySide6
- `src/Core.py`: Central controller managing camera, sampling, and worker threads
- `src/Workers.py`: Background worker threads for frame processing and sampling
- `src/DataClasses.py`: Data structures for samples and frame data

**GUI Components:**
- `src/Widgets.py`: Custom Qt widgets (AnalyserWidget, Graph, PixmapWidget, TableUnit)
- Main window provides sensor feed, analyzer display, sampling controls, and plotting

**Utilities:**
- `src/utils.py`: Utility functions and unit conversion helpers
- `src/curves.py`: Gaussian curve fitting algorithms
- `src/tooltips.py`: UI tooltip definitions
- `src/cycle.py`: Cyclic measurement functionality
- `src/s_server.py`: Socket server for remote control
- `src/client.py`: Socket client functionality

**Key Features:**
- Real-time camera feed processing with gaussian curve fitting
- Multi-threaded architecture with QThread workers
- Socket server for remote control integration
- Cyclic measurement functionality

### Threading Architecture

The application uses a multi-threaded design:
- Main GUI thread for user interface
- `FrameWorker` thread for real-time camera frame processing
- `SampleWorker` thread for taking multiple subsamples and statistical analysis
- Workers communicate via Qt signals/slots

### Data Flow

1. Camera frames → `Core.onFramePassedFromCamera()` → `FrameWorker`
2. Frame analysis → histogram generation → gaussian curve fitting → center detection
3. Sample collection → statistical processing (outlier removal, smoothing) → measurement calculation
4. Results → GUI updates and optional socket communication

## Project Structure

The codebase has been refactored with LinuxCNC remote driver components moved to a separate location. The main application now focuses purely on the webcam sensor measurement functionality.

## Package Management

The project has simplified dependency management:
- `requirements.txt`: Contains both runtime and development dependencies
- Development tools include: pytest, pytest-qt, coverage, pre-commit
- No separate requirements-dev.txt file needed

## Testing Structure

Tests are located in `tests/` directory with modules corresponding to source files. The project includes both unit tests and Qt-specific tests using pytest-qt.

## Development Notes

- The project is currently on the `major-refactoring` branch
- LinuxCNC remote driver functionality has been separated from the main codebase
- Development dependencies are now consolidated into the main requirements.txt file
- Pre-commit hooks are available for code quality enforcement