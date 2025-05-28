# Neural-Decoder Architecture Documentation

## Project Overview

Neural-Decoder is a real-time EEG alpha wave detection system designed for brain-computer interface applications. The system can classify eye states (open vs. closed) based on alpha wave activity in the 8-12 Hz frequency range.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hardware      │    │   Software      │    │   User          │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ OpenBCI     │ │    │ │ Data        │ │    │ │ Real-time   │ │
│ │ Ganglion    │◄┼────┼►│ Collection  │ │    │ │ Feedback    │ │
│ │ Board       │ │    │ │             │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │        │        │    │        ▲        │
│ ┌─────────────┐ │    │        ▼        │    │        │        │
│ │ EEG         │ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Electrodes  │ │    │ │ Signal      │ │    │ │ Visualiza-  │ │
│ │             │ │    │ │ Processing  │ │    │ │ tion        │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    │        │        │    └─────────────────┘
                       │        ▼        │
                       │ ┌─────────────┐ │
                       │ │ Machine     │ │
                       │ │ Learning    │ │
                       │ │ Models      │ │
                       │ └─────────────┘ │
                       └─────────────────┘
```

### Data Flow Pipeline

```
Raw EEG Signal
      │
      ▼
┌─────────────┐
│ Bluetooth   │
│ Acquisition │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Filtering   │
│ & Denoising │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Feature     │
│ Extraction  │
└─────────────┘
      │
      ▼
┌─────────────┐
│ ML Model    │
│ Inference   │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Real-time   │
│ Output      │
└─────────────┘
```

## Project Structure

### Directory Organization

```
Neural-Decoder/
├── src/                          # Core source code
│   ├── data_collection/          # EEG data acquisition modules
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data preprocessing utilities
│   ├── signal_processing/        # Signal processing algorithms
│   │   ├── __init__.py
│   │   ├── signal_processing.py  # Core signal processing functions
│   │   ├── welch_method.py       # Frequency analysis
│   │   └── filters.py           # Digital filtering
│   ├── models/                   # Machine learning models
│   │   ├── __init__.py
│   │   ├── neural_network.py     # Traditional ML approach
│   │   ├── realtime_predictor.py # Advanced neural network
│   │   └── model_training.py     # Training utilities
│   ├── visualization/            # Real-time plotting and analysis
│   │   ├── __init__.py
│   │   ├── visualize_data.py     # Data visualization utilities
│   │   └── spectrogram_analysis.py # Frequency domain visualization
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── config_loader.py      # Configuration management
├── scripts/                      # Main execution scripts
│   ├── train_model.py           # Training script
│   ├── realtime_detection.py    # Real-time inference
│   └── alpha_visualization.py   # Live visualization
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage
│   ├── raw/                     # Raw EEG recordings
│   ├── processed/               # Processed data
│   └── exports/                 # Exported data for analysis
├── models/                       # Trained model storage
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # This file
│   ├── Images/                  # Documentation images
│   └── training_plots/          # Training visualization
├── examples/                     # Example scripts and tutorials
├── tests/                        # Unit tests
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
└── LICENSE                      # MIT license
```

## Core Components

### 1. Data Collection (`src/data_collection/`)

**Purpose**: Interface with OpenBCI Ganglion board for real-time EEG acquisition.

**Key Functions**:
- Bluetooth connection management
- Real-time data streaming
- Data buffering and windowing
- Basic preprocessing

**Main Files**:
- `preprocessing.py`: Raw data extraction and format conversion

### 2. Signal Processing (`src/signal_processing/`)

**Purpose**: Clean and extract features from raw EEG signals.

**Processing Pipeline**:
1. **Detrending**: Remove DC offset and linear trends
2. **Bandpass Filtering**: 3-45 Hz to remove artifacts
3. **Notch Filtering**: 58-62 Hz to remove power line noise
4. **Normalization**: Scale signals to [-1, 1] range
5. **Feature Extraction**: Alpha power, spectral features, Hjorth parameters

**Main Files**:
- `signal_processing.py`: Core filtering and processing functions
- `welch_method.py`: Frequency domain analysis using Welch's method
- `filters.py`: Digital filter implementations

### 3. Machine Learning Models (`src/models/`)

**Purpose**: Classify eye states based on processed EEG features.

#### Traditional Approach (`neural_network.py`)
- Feature-based classification
- Dense neural network with dropout
- Uses extracted features (alpha power, RMS, variance, etc.)

#### Advanced Approach (`realtime_predictor.py`)
- Hybrid time-frequency domain architecture
- **Time Domain Path**: Bidirectional LSTM with attention mechanism
- **Frequency Domain Path**: Dual residual RNNs
- **Fusion Layer**: Concatenated features through residual dense layers

**Model Architecture**:
```
Time Domain (600 samples)     Frequency Domain (spectral features)
        │                                     │
        ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│ Residual BiLSTM │                 │ Residual RNN 1  │
│ (bidirectional) │                 └─────────────────┘
└─────────────────┘                           │
        │                                     ▼
        ▼                           ┌─────────────────┐
┌─────────────────┐                 │ Residual RNN 2  │
│ Attention Layer │                 └─────────────────┘
│ (second-half    │                           │
│  masking)       │                           ▼
└─────────────────┘                 ┌─────────────────┐
        │                           │ Output Vector   │
        ▼                           └─────────────────┘
┌─────────────────┐                           │
│ Attended Vector │                           │
└─────────────────┘                           │
        │                                     │
        └─────────────┬───────────────────────┘
                      ▼
              ┌─────────────────┐
              │ Concatenation   │
              └─────────────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Residual Dense  │
              │ Layers          │
              └─────────────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Binary Output   │
              │ (Open/Closed)   │
              └─────────────────┘
```

### 4. Visualization (`src/visualization/`)

**Purpose**: Real-time monitoring and analysis of EEG signals and model outputs.

**Features**:
- Live EEG waveform display
- Real-time alpha power monitoring
- Threshold-based detection status
- Training progress visualization

### 5. Configuration Management (`src/utils/`)

**Purpose**: Centralized configuration handling for easy customization.

**Features**:
- YAML-based configuration
- Environment-specific settings
- Default value fallbacks
- Dot-notation access to nested config values

## Real-Time Processing

### Sliding Window Approach

The system uses a sliding window approach for real-time processing:

- **Window Size**: 3 seconds (600 samples at 200 Hz)
- **Update Rate**: Every 1.5 seconds (50% overlap)
- **Buffer Management**: Continuous data buffer with windowed processing

### Latency Considerations

- **Hardware Latency**: ~10-50ms (Bluetooth + ADC)
- **Processing Latency**: ~100-200ms (filtering + feature extraction)
- **Model Inference**: ~50-100ms (neural network forward pass)
- **Total System Latency**: ~200-400ms

## Performance Characteristics

### Model Performance
- **Accuracy**: 85-95% (depending on user and conditions)
- **False Positive Rate**: <10%
- **False Negative Rate**: <15%
- **Training Time**: 5-10 minutes for basic model

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB for models and data
- **OS**: Windows, macOS, Linux (Python 3.8+)

## Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
# Device settings
device:
  mac_address: "f0:17:3b:41:ec:7d"
  sampling_rate: 200
  
# Signal processing parameters
signal_processing:
  window_size_sec: 3.0
  overlap_percent: 0.5
  alpha_band:
    low_freq: 8.0
    high_freq: 12.0
    
# Machine learning parameters
ml:
  training:
    epochs: 20
    batch_size: 32
```

## Extension Points

### Adding New Features
1. **New Signal Processing**: Add functions to `src/signal_processing/`
2. **Custom Models**: Implement in `src/models/`
3. **Additional Visualizations**: Extend `src/visualization/`
4. **New Hardware**: Modify `src/data_collection/`

### Model Customization
- Modify hyperparameters in `configs/config.yaml`
- Implement custom architectures in `src/models/`
- Add new feature extraction methods in `src/signal_processing/`

## Future Enhancements

### Planned Features
1. **Multi-channel Support**: Process multiple EEG channels simultaneously
2. **Advanced Models**: Transformer-based architectures
3. **Real-time Adaptation**: Online learning capabilities
4. **Mobile Support**: Android/iOS applications
5. **Cloud Integration**: Remote model training and deployment

### Research Directions
1. **Artifact Rejection**: Automatic detection and removal of eye blinks, muscle artifacts
2. **Personalization**: User-specific model adaptation
3. **Multi-modal**: Integration with other biosignals (EOG, EMG)
4. **Closed-loop**: Real-time neurofeedback applications 