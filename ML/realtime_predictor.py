import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import pandas as pd
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import time
"""
 Time-Domain Path                                    Frequency-Domain Path
 ───────────────                                    ──────────────────────
 
 (batch_size x input_size)                           (batch_size x n_freq_features)
          |                                                      |
          | transpose to (batch_size x seq_len x 1)             |
          v                                                      v
 +----------------------+                             +----------------------+
 |  Residual BiLSTM    |                             |  Residual RNN (1)    |
 |  (bidirectional)    |                             +----------------------+
 +----------------------+                                      |
          |                                                    v
          v                                          +----------------------+
 +----------------------+                             |  Residual RNN (2)    |
 |  Attention (w/      |                             +----------------------+
 |  second-half mask)  |                                      |
 +----------------------+                                  (take last hidden)
          |                                                    v
     (weighted sum)                                   +----------------------+
          v                                           |  RNN Output Vector   |
 +----------------------+                             +----------------------+
 |  Attended Vector     |
 +----------------------+

                 ┌──────────────────────────── Concatenate ────────────────────────────┐
                 │    [Attended Vector, RNN Output Vector] (feature combination)       │
                 └──────────────────────────────────────────────────────────────────────┘
                                                      |
                                                      v
                                          +-----------------------------+
                                          |  Residual Dense (1)        |
                                          +-----------------------------+
                                                      |
                                                      v
                                          +-----------------------------+
                                          |  Residual Dense (2)        |
                                          +-----------------------------+
                                                      |
                                                      v
                                          +-----------------------------+
                                          |  Linear(2) (2-class output)|
                                          +-----------------------------+
                                                      |
                                                      v
                                          +-----------------------------+
                                          |       Final Output         |
                                          +-----------------------------+"""




class EEGPreprocessor:
    def __init__(self, fs=200):
        # Normalization parameters
        self.train_mean = None
        self.train_std = None
        
        # Filter parameters
        self.fs = fs
        self.stride = 300
        self.filter_buffer = []
        self.min_filter_samples = int(fs * 0.5)  # 0.5 seconds minimum
        
        # Define EEG frequency bands
        self.freq_bands = {
            'alpha': (8, 13),   # Alpha: 8-13 Hz
        }
        
        # Create filters
        self.setup_filters()
        
    def setup_filters(self):
        """Initialize bandpass and notch filters"""
        # Bandpass filter (1-50 Hz)
        nyquist = self.fs / 2
        low = 3 / nyquist
        high = 30 / nyquist
        self.b_bandpass, self.a_bandpass = signal.butter(4, [low, high], btype='band')
        
        # Notch filter (60 Hz)
        Q = 30  # Quality factor
        w0 = 60 / nyquist
        self.b_notch, self.a_notch = signal.iirnotch(w0, Q)
    
    def compute_stft_features(self, window):
        """Compute STFT-based features for real-time processing"""
        # Parameters for STFT
        nperseg = min(600, len(window))  # Length of each segment
        noverlap = nperseg // 2         # 50% overlap
        
        # Compute STFT
        f, t, Zxx = signal.stft(window, fs=self.fs, nperseg=nperseg, 
                               noverlap=noverlap, boundary=None)
        
        # Extract power in each frequency band
        band_powers = {}
        for band_name, (low, high) in self.freq_bands.items():
            # Find frequencies within band
            mask = (f >= low) & (f <= high)
            # Compute average power in band
            band_power = np.mean(np.abs(Zxx[mask, -1])**2)  # Use only the latest time point
            band_powers[band_name] = band_power
            
        return band_powers
    
    def compute_wavelet_features(self, window):
        """Compute wavelet-based features for real-time processing"""
        # Compute wavelet transform for each band
        band_powers = {}
        for band_name, (low, high) in self.freq_bands.items():
            # Calculate center frequency for the band
            center_freq = (low + high) / 2
            # Number of cycles in the wavelet
            num_cycles = 6
            
            # Create Morlet wavelet
            width = num_cycles / (2 * np.pi * center_freq)
            wavelet = signal.morlet2(len(window), center_freq/self.fs, width)
            
            # Convolve signal with wavelet
            power = np.abs(signal.convolve(window, wavelet, mode='same'))**2
            band_powers[band_name] = np.mean(power)
            
        return band_powers
        
    def fit(self, training_data):
        """Calculate normalization parameters from training data"""
        self.train_mean = np.mean(training_data)
        self.train_std = np.std(training_data)
        print(f"Training stats - Mean: {self.train_mean:.2f}, Std: {self.train_std:.2f}")
        
    def normalize_point(self, point):
        """Normalize a single point using training statistics"""
        return (point - self.train_mean) / self.train_std
    
    def apply_filters(self, data):
        """Apply bandpass and notch filters to data"""
        filtered = signal.filtfilt(self.b_bandpass, self.a_bandpass, data)
        filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered)
        return filtered
    
    def preprocess_point(self, new_point):
        """Process a single new point with frequency decomposition"""
        self.filter_buffer.append(new_point)
        
        if len(self.filter_buffer) >= self.min_filter_samples:
            # Apply filters to buffer
            filtered = self.apply_filters(np.array(self.filter_buffer))
            
            # Get the latest filtered point
            filtered_point = filtered[-1]
            
            # Compute frequency features
            freq_features = self.compute_stft_features(filtered)
            
            # Normalize using training statistics
            normalized_point = self.normalize_point(filtered_point)
            
            # Keep buffer size manageable
            if len(self.filter_buffer) > self.min_filter_samples:
                self.filter_buffer.pop(0)
                
            return normalized_point, freq_features
        
        # If not enough samples, just normalize
        return self.normalize_point(new_point), None

class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        # Projection layer if input and output sizes don't match
        self.proj = nn.Linear(input_size, hidden_size * 2) if bidirectional else nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
    
    def forward(self, x, hc=None):
        identity = self.proj(x)  # Project input to match LSTM output size
        out, (h, c) = self.lstm(x, hc)
        out = out + identity  # Residual connection
        out = self.layer_norm(out)  # Layer normalization
        return out, (h, c)

class ResidualRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity='relu'
        )
        self.proj = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h=None):
        identity = self.proj(x)
        out, h = self.rnn(x, h)
        out = out + identity
        out = self.layer_norm(out)
        return out, h

class ResidualDense(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    
    def forward(self, x):
        identity = self.proj(x)
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + identity
        out = self.layer_norm(out)
        return out

class EEGClassifier(nn.Module):
    def __init__(self, input_size=600, n_freq_features=1, hidden_size=128, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Residual LSTM for time-domain features
        self.lstm = ResidualLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.3
        )
        
        # Residual RNN layers for frequency features
        self.rnn_layers = nn.ModuleList([
            ResidualRNN(
                input_size=n_freq_features if i == 0 else hidden_size//2,
                hidden_size=hidden_size//2
            ) for i in range(2)
        ])
        
        # Attention mechanism with second-half bias
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Residual dense layers
        lstm_output_size = hidden_size * 2  # bidirectional LSTM
        self.fc_layers = nn.Sequential(
            ResidualDense(lstm_output_size + hidden_size//2, hidden_size, dropout=0.4),
            ResidualDense(hidden_size, hidden_size//2, dropout=0.3),
            nn.Linear(hidden_size//2, 2)  # Final classification layer
        )
    
    def forward(self, x, freq_features):
        # Reshape for LSTM: (batch_size, sequence_length, 1)
        x = x.transpose(1, 2)
        
        # Initialize hidden states for LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Process time-domain features through Residual LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Create attention mask to focus on second half
        seq_len = lstm_out.size(1)
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        
        # Create and apply second-half bias mask
        mask = torch.ones_like(attention_scores)
        mask[:, :seq_len//2] = 0.0  # Zero out first half
        attention_scores = attention_scores * mask
        
        # Apply softmax after masking
        attention_weights = F.softmax(attention_scores, dim=1)
        lstm_attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Process frequency features through Residual RNN layers
        rnn_out = freq_features.unsqueeze(1)
        for rnn_layer in self.rnn_layers:
            rnn_out, _ = rnn_layer(rnn_out)
        rnn_out = rnn_out[:, -1, :]
        
        # Combine features and pass through residual dense layers
        combined = torch.cat([lstm_attended, rnn_out], dim=1)
        output = self.fc_layers(combined)
        
        return output

class RealTimePredictor:
    def __init__(self, window_size=600, fs=200, stride=50):
        self.window_size = window_size
        self.fs = fs
        self.stride = stride  # Number of points to slide window
        
        # Initialize preprocessor and model
        self.preprocessor = EEGPreprocessor(fs=fs)
        self.model = EEGClassifier(input_size=window_size, n_freq_features=1)
        
        # Buffer for incoming data
        self.data_buffer = []
        self.points_since_last_inference = 0
        self.last_prediction = None
        
        # Create directory for plots
        self.plot_dir = 'training_plots'
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(f'runs/eeg_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    def add_points(self, new_points):
        """Add points to the buffer and process when we have exactly 300 new points"""
        self.data_buffer.extend(new_points)
        self.points_since_last_inference += len(new_points)
        
        # Process only when we have exactly 300 new points
        if self.points_since_last_inference >= 300:
            self._process_buffer()
            self.points_since_last_inference = 0
    
    def _process_buffer(self):
        """Process the current buffer and make a prediction"""
        if len(self.data_buffer) < self.window_size:
            return None, 0.0
            
        # Get the latest window of data
        window = np.array(self.data_buffer[-self.window_size:])
        
        # Process window
        processed_window = self.preprocessor.apply_filters(window)
        normalized_window = np.array([self.preprocessor.normalize_point(p) for p in processed_window])
        freq_features = self.preprocessor.compute_stft_features(processed_window)
        
        # Convert to tensors
        X = torch.FloatTensor(normalized_window).unsqueeze(0).unsqueeze(0)
        freq_tensor = torch.FloatTensor([[v for v in freq_features.values()]])
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(X, freq_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Store prediction
        self.last_prediction = (prediction, confidence, time.time())
        return prediction, confidence
    
    def get_latest_prediction(self):
        """Get the most recent prediction"""
        return self.last_prediction

    def plot_training_metrics(self, metrics):
        """Plot and save training metrics"""
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['loss'], 'b-', label='Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.plot_dir}/loss_curve.png')
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['accuracy'], 'g-', label='Accuracy')
        plt.title('Training Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.plot_dir}/accuracy_curve.png')
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{self.plot_dir}/confusion_matrix.png')
        plt.close()
        
        # Save classification report
        with open(f'{self.plot_dir}/classification_report.txt', 'w') as f:
            f.write(metrics['classification_report'])
    
    def train(self, training_data, labels, epochs=10, batch_size=32):
        """Train the model with the given data"""
        # Store metrics
        training_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'all_predictions': [],
            'all_labels': []
        }
        
        # Fit preprocessor with training data
        self.preprocessor.fit(training_data)
        
        # Preprocess all training data
        processed_data = []
        processed_freq = []
        processed_labels = []
        
        for i in range(0, len(training_data) - self.window_size + 1, self.preprocessor.stride):
            window = training_data[i:i + self.window_size]
            processed_window = self.preprocessor.apply_filters(window)
            normalized_window = [self.preprocessor.normalize_point(p) for p in processed_window]
            freq_features = self.preprocessor.compute_stft_features(processed_window)
            
            # Get the label for this window (use majority vote)
            window_labels = labels[i:i + self.window_size]
            window_label = int(np.mean(window_labels) > 0.5)
            
            processed_data.append(normalized_window)
            processed_freq.append([v for v in freq_features.values()])
            processed_labels.append(window_label)
        
        processed_data = np.array(processed_data)
        processed_freq = np.array(processed_freq)
        processed_labels = np.array(processed_labels)
        
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed freq shape: {processed_freq.shape}")
        print(f"Processed labels shape: {processed_labels.shape}")
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(processed_data).unsqueeze(1)
        freq_features = torch.FloatTensor(processed_freq)
        y = torch.LongTensor(processed_labels)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, freq_features, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Log model graph
        dummy_input = (torch.randn(1, 1, self.window_size), torch.randn(1, 1))
        self.writer.add_graph(self.model, dummy_input)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0
            epoch_predictions = []
            epoch_labels = []
            
            for batch_X, batch_freq, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X, batch_freq)
                loss = criterion(outputs, batch_y)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == batch_y).sum().item() / batch_y.size(0)
                
                loss.backward()
                optimizer.step()
                
                # Store predictions and labels
                epoch_predictions.extend(predicted.cpu().numpy())
                epoch_labels.extend(batch_y.cpu().numpy())
                
                # Accumulate metrics
                epoch_loss += loss.item()
                epoch_acc += acc
                n_batches += 1
            
            # Calculate average metrics
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
            
            # Store metrics
            training_metrics['epoch'].append(epoch + 1)
            training_metrics['loss'].append(avg_loss)
            training_metrics['accuracy'].append(avg_acc)
            training_metrics['all_predictions'].extend(epoch_predictions)
            training_metrics['all_labels'].extend(epoch_labels)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Accuracy/train', avg_acc, epoch)
            
            # Log model weights histograms
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            
            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')
        
        # Calculate final metrics
        final_predictions = np.array(training_metrics['all_predictions'])
        final_labels = np.array(training_metrics['all_labels'])
        
        training_metrics['confusion_matrix'] = confusion_matrix(final_labels, final_predictions)
        training_metrics['classification_report'] = classification_report(final_labels, final_predictions)
        
        # Plot metrics
        self.plot_training_metrics(training_metrics)
        
        self.writer.close()
        print(f"\nTraining plots saved in {self.plot_dir}/")
        print("\nClassification Report:")
        print(training_metrics['classification_report'])

    def export_model(self):
        """Save the trained model"""
        torch.save(self.model.state_dict(), 'real_time_model.pth')
    
    def predict_point(self, new_point):
        """Make prediction for a single new point"""
        # Add the point to buffer
        self.add_points(new_point)
        
        # Return the last prediction if we have one
        return self.get_latest_prediction()[0:2] if self.get_latest_prediction() else (None, 0)

def convert_number_to_label(number):
    if number == 0:
        return "Eyes Open"
    return "Eyes Closed"

if __name__ == "__main__":
    # Load data and create predictor
    data = pd.read_csv("extracted_data.csv")
    channel0 = data['EXG Channel 0'].values
    
    # Create labels (1 for eyes closed, 0 for eyes open)
    labels = []
    for i in range(len(channel0)):
        time_in_seconds = i//200
        if 30<=time_in_seconds<60 or 90<=time_in_seconds<120:
            labels.append(1)
        else:
            labels.append(0)    
    labels = np.array(labels)
    
    # Create and train predictor
    predictor = RealTimePredictor()
    predictor.train(channel0, labels, epochs=20)
    predictor.export_model()
    
    # Test predictions on original data but reversed
    test_channel0 = channel0[::-1]
    test_labels = labels[::-1]
    
    predictions = []
    confidences = []
    inference_times = []
    total_processing_time = 0
    
    # Make predictions in batches and measure time
    print("\nStarting inference timing test...")
    batch_size = 300
    for i in range(0, len(test_channel0), batch_size):
        batch = test_channel0[i:i + batch_size]
        
        start_time = time.perf_counter()
        predictor.add_points(batch)
        end_time = time.perf_counter()
        
        # Get the latest prediction after adding points
        result = predictor.get_latest_prediction()
        if result:
            pred, conf, _ = result
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Buffer size: {len(predictor.data_buffer)}")
            print(f"Data Point: {i+len(batch)}/{len(test_channel0)}, "
                  f"Prediction: {convert_number_to_label(pred)}, "
                  f"Confidence: {conf:.2f}, "
                  f"Label: {convert_number_to_label(test_labels[i])}, "
                  f"Inference Time: {inference_time:.2f}ms")
            
            predictions.append(pred)
            confidences.append(conf)
            inference_times.append(inference_time)
            total_processing_time += inference_time
    
    # Calculate accuracy on predictions
    if predictions:
        test_labels = test_labels[-len(predictions):]  # Align labels with predictions
        accuracy = np.mean(np.array(predictions) == test_labels)
        print(f"\nTest Accuracy: {accuracy:.2f}")
        
        # Plot confidence over time
        plt.figure(figsize=(10, 6))
        plt.plot(confidences)
        plt.title('Prediction Confidence Over Time')
        plt.xlabel('Prediction Number')
        plt.ylabel('Confidence')
        plt.grid(True)
        plt.savefig('prediction_confidence.png')
        plt.close()
