# Neural-Decoder Setup Guide

This guide will help you set up the Neural-Decoder system for real-time EEG alpha wave detection.

## Hardware Requirements

### OpenBCI Ganglion Board
- **Required**: OpenBCI Ganglion EEG board
- **Bluetooth**: Built-in Bluetooth Low Energy (BLE)
- **Channels**: 4 analog input channels (we use channel 0)
- **Sampling Rate**: 200 Hz

### EEG Electrodes
- **Type**: Dry or wet EEG electrodes
- **Placement**: Single channel setup (typically O1 or Oz position)
- **Reference**: Earlobe or mastoid reference
- **Ground**: Forehead or other neutral position

### Computer Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **CPU**: Multi-core processor (Intel i5 or equivalent)
- **RAM**: 4GB minimum, 8GB recommended
- **Bluetooth**: Bluetooth 4.0+ for Ganglion connectivity
- **Python**: 3.8 or higher

## Software Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Neural-Decoder.git
cd Neural-Decoder
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n neural-decoder python=3.8
conda activate neural-decoder

# Using venv
python -m venv neural-decoder
source neural-decoder/bin/activate  # On Windows: neural-decoder\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import brainflow; print('BrainFlow installed successfully')"
python -c "import torch; print('PyTorch installed successfully')"
python -c "import tensorflow; print('TensorFlow installed successfully')"
```

## Hardware Setup

### 1. OpenBCI Ganglion Preparation

1. **Charge the Board**: Ensure the Ganglion is fully charged
2. **Power On**: Press and hold the power button until the LED turns on
3. **Note MAC Address**: Record the Bluetooth MAC address (format: `XX:XX:XX:XX:XX:XX`)

### 2. Electrode Placement

For alpha wave detection, we recommend:

```
     Fp1 ● ● Fp2
        ●   ●
     C3 ● ● ● C4
        ●   ●
     P3 ● ● ● P4
        ●   ●
     O1 ● [●] O2  <- Primary electrode (Channel 0)
```

**Standard Setup**:
- **Active Electrode**: O1 (left occipital) → Channel 0
- **Reference**: Left earlobe → Channel 1  
- **Ground**: Forehead → BIAS pin
- **Channel 2 & 3**: Can be left unconnected

### 3. Connection Guide

1. **Clean Skin**: Use alcohol swabs to clean electrode sites
2. **Apply Electrodes**: 
   - Dry electrodes: Apply with gentle pressure
   - Wet electrodes: Apply electrode gel first
3. **Connect to Ganglion**:
   - Red wire → Channel 0 (active)
   - Black wire → Channel 1 (reference)  
   - White wire → BIAS (ground)
4. **Check Impedance**: Ensure good contact (< 50kΩ)

## Configuration

### 1. Update Device MAC Address

Edit `configs/config.yaml`:

```yaml
device:
  mac_address: "XX:XX:XX:XX:XX:XX"  # Replace with your Ganglion's MAC address
```

### 2. Customize Settings (Optional)

```yaml
# Signal processing parameters
signal_processing:
  window_size_sec: 3.0      # Analysis window size
  alpha_band:
    low_freq: 8.0           # Alpha band lower bound
    high_freq: 12.0         # Alpha band upper bound

# Training parameters  
ml:
  training:
    epochs: 20              # Training epochs
    batch_size: 32          # Batch size for training
```

## Quick Start

### 1. Test Hardware Connection

```bash
python examples/bluetooth-stream-test.py
```

This should display live EEG data if your hardware is connected properly.

### 2. Collect Training Data

```bash
python scripts/train_model.py
```

Follow the prompts to:
1. Press ENTER for "Eyes Open" segments (15 seconds each)
2. Press ENTER for "Eyes Closed" segments (15 seconds each)
3. Repeat for 4 total segments

### 3. Run Real-time Detection

```bash
python scripts/realtime_detection.py
```

### 4. Launch Visualization

```bash
python scripts/alpha_visualization.py
```

## Troubleshooting

### Common Issues

#### 1. Bluetooth Connection Failed

**Error**: `Unable to connect to device`

**Solutions**:
- Verify MAC address in `configs/config.yaml`
- Ensure Ganglion is powered on and discoverable
- Try restarting Bluetooth on your computer
- Move closer to the device (within 10 feet)

```bash
# Test Bluetooth connection
python -c "
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
params = BrainFlowInputParams()
params.mac_address = 'YOUR_MAC_ADDRESS'
board = BoardShim(BoardIds.GANGLION_NATIVE_BOARD.value, params)
board.prepare_session()
print('Connection successful!')
board.release_session()
"
```

#### 2. Poor Signal Quality

**Symptoms**: Noisy or flat EEG signals

**Solutions**:
- Clean electrode sites with alcohol
- Ensure good electrode contact
- Check impedance levels
- Avoid electrical interference (phones, WiFi)
- Sit still during recording

#### 3. Import Errors

**Error**: `ModuleNotFoundError`

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Configuration Errors

**Error**: `Configuration file not found`

**Solutions**:
- Ensure you're running scripts from the project root directory
- Check that `configs/config.yaml` exists
- Verify YAML syntax is correct

### Performance Optimization

#### 1. Reduce Latency
- Close unnecessary applications
- Use a dedicated USB port for Bluetooth adapter
- Disable WiFi if not needed
- Run with elevated priority:

```bash
# Linux/macOS
sudo nice -n -10 python scripts/realtime_detection.py

# Windows (run as administrator)
python scripts/realtime_detection.py
```

#### 2. Improve Accuracy
- Ensure consistent electrode placement
- Minimize head movement during recording
- Train in similar conditions to usage
- Collect more training data

## Advanced Configuration

### Custom Electrode Montage

For multi-channel recording, modify `configs/config.yaml`:

```yaml
device:
  channels: [0, 1, 2, 3]  # Use all 4 channels
  
signal_processing:
  channel_names: ["O1", "O2", "P3", "P4"]
  reference_channels: [1]  # Use channel 1 as reference
```

### Custom Frequency Bands

```yaml
signal_processing:
  frequency_bands:
    delta: [1, 4]
    theta: [4, 8]
    alpha: [8, 12]
    beta: [12, 30]
    gamma: [30, 50]
```

### Model Hyperparameters

```yaml
ml:
  advanced:
    hidden_size: 256        # Increase for more complex models
    num_layers: 4           # Deeper networks
    dropout: 0.3            # Regularization
    
  training:
    learning_rate: 0.0001   # Lower for stable training
    early_stopping_patience: 5
```

## Data Collection Best Practices

### 1. Session Preparation
- Ensure participant is comfortable and relaxed
- Explain the task clearly
- Test electrode connections beforehand
- Have the participant practice eye opening/closing

### 2. Recording Environment
- Quiet room with minimal distractions
- Consistent lighting
- Comfortable seating
- No electronic devices nearby

### 3. Data Quality Checks
- Monitor signal quality in real-time
- Check for artifacts (eye blinks, muscle tension)
- Ensure consistent baseline
- Record multiple sessions for robust training

### 4. Training Tips
- **Clear Instructions**: "Open/close eyes naturally"
- **Consistent Timing**: Use beeps or verbal cues
- **Multiple Sessions**: Train across different times/days
- **Validation**: Test with new data before deployment

## Next Steps

Once you have the system running:

1. **Experiment with Parameters**: Try different alpha band frequencies
2. **Collect More Data**: Additional training improves accuracy
3. **Explore Applications**: Build neurofeedback applications
4. **Contribute**: Share improvements with the community

For more advanced topics, see:
- [Architecture Documentation](ARCHITECTURE.md)
- [Examples Directory](../examples/)
- [API Reference](API.md)

## Support

If you encounter issues:

1. **Check Logs**: Look for error messages in the console
2. **Hardware Test**: Verify with OpenBCI GUI first
3. **Documentation**: Review this guide and architecture docs
4. **Community**: Post issues on GitHub with detailed error logs
5. **Contact**: Reach out to the maintainers

Happy brain-computer interfacing! 🧠⚡ 