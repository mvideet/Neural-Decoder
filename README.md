# Neural-Decoder: Real-Time EEG Alpha Wave Detection System

A comprehensive EEG analysis system for real-time alpha brain wave detection using machine learning and signal processing. The system can detect when a user's eyes are open vs. closed based on alpha wave activity (8-12 Hz).

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

## 🧠 Features

- **Real-time EEG Processing**: Live signal acquisition from OpenBCI Ganglion board
- **Advanced Signal Processing**: Bandpass filtering, notch filtering, and normalization
- **Multiple ML Approaches**: Both traditional feature-based and deep learning models
- **Real-time Visualization**: Live EEG waveforms and alpha power monitoring
- **Hybrid Neural Architecture**: Combines time-domain and frequency-domain processing
- **Threshold Optimization**: Automatic threshold tuning for optimal classification

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Hardware Setup

1. OpenBCI Ganglion EEG board
2. Bluetooth connection to computer
3. EEG electrodes properly positioned

### Basic Usage

1. **Data Collection and Training**:
```bash
python scripts/train_model.py
```

2. **Real-time Alpha Detection**:
```bash
python scripts/realtime_detection.py
```

3. **Live Visualization**:
```bash
python scripts/alpha_visualization.py
```

## 📁 Project Structure

```
Neural-Decoder/
├── src/                          # Source code
│   ├── data_collection/          # EEG data acquisition
│   ├── signal_processing/        # Signal processing utilities
│   ├── models/                   # Machine learning models
│   ├── visualization/            # Real-time plotting and analysis
│   └── utils/                    # Utility functions
├── scripts/                      # Main execution scripts
├── data/                         # Data storage
│   ├── raw/                      # Raw EEG recordings
│   ├── processed/                # Filtered and processed data
│   └── exports/                  # Exported data for analysis
├── models/                       # Trained model storage
├── configs/                      # Configuration files
├── docs/                         # Documentation
├── tests/                        # Unit tests
└── examples/                     # Example usage scripts
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:
- EEG device settings (MAC address, sampling rate)
- Signal processing parameters
- Model hyperparameters
- Visualization settings

## 📊 How It Works

### 1. Signal Processing Pipeline
```
Raw EEG → Detrending → Bandpass Filter → Notch Filter → Normalization → Feature Extraction
```

### 2. Feature Extraction
- Alpha power (8-12 Hz) using Welch's method
- Relative alpha power
- Alpha peak frequency
- RMS amplitude and variance
- Hjorth mobility and complexity

### 3. Machine Learning Models
- **Traditional**: Feature-based neural network
- **Advanced**: Hybrid architecture with BiLSTM and attention
- **Real-time**: Sliding window processing with 600-sample windows

## 🎯 Applications

- **Neurofeedback Training**: Monitor relaxation states
- **Meditation Apps**: Real-time meditation feedback
- **Brain-Computer Interfaces**: Simple binary control
- **Research**: EEG signal analysis and alpha wave studies
- **Assistive Technology**: Eye-state detection

## 📈 Performance

- **Sampling Rate**: 200 Hz
- **Latency**: ~1.5 seconds
- **Window Size**: 3 seconds (600 samples)
- **Update Rate**: Every 1.5 seconds (50% overlap)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenBCI for the Ganglion EEG hardware
- BrainFlow library for EEG data acquisition
- The neuroscience and BCI community

## 📞 Contact

Videet Mehta - [GitHub](https://github.com/your-username)

Project Link: [https://github.com/your-username/Neural-Decoder](https://github.com/your-username/Neural-Decoder) 