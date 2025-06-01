import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import time
import plotly.graph_objects as go
from tqdm import tqdm
import gc
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, length = x.size()

        # Reshape for attention computation
        proj_query = self.query(x).permute(0, 2, 1)  # B x L x C/8
        proj_key = self.key(x)  # B x C/8 x L
        energy = torch.bmm(proj_query, proj_key)  # B x L x L
        attention = self.softmax(energy)  # B x L x L

        proj_value = self.value(x).permute(0, 2, 1)  # B x L x C
        out = torch.bmm(attention, proj_value)  # B x L x C
        out = out.permute(0, 2, 1)  # B x C x L

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], dilation_rates=[1, 2, 4, 8], embed_dim=256, num_heads=8, num_layers=2):
        super(UNet1D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        for feature, dilation in zip(features, dilation_rates):
            self.downs.append(
                self._multi_scale_conv(in_channels, feature, dilation)
            )
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(
                self._double_conv(feature * 2, feature)
            )

        bottleneck_channels = features[-1] * 2
        self.bottleneck_conv = self._double_conv(features[-1], bottleneck_channels)
        
        # Self-attention module in the bottleneck
        self.attention = SelfAttention1D(bottleneck_channels)

        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _multi_scale_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck_conv(x)
        
        # Apply attention in the bottleneck
        x = self.attention(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                diff = skip_connection.size()[2] - x.size()[2]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    skip_connection = F.pad(skip_connection, (0, -diff))

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.sigmoid(self.final_conv(x))

def process_large_2d_array(input_file, output_file=None, dtype=np.float32, row_chunk=100, apply_filter=True, min_val=-800000, max_val=800000):
    if input_file.endswith('.npy'):
        array_shape = np.lib.format.open_memmap(input_file).shape
        print(f"Processing array with shape: {array_shape}")
    elif input_file.endswith('.csv'):
        with open(input_file, 'r') as f:
            for i, _ in enumerate(f):
                pass
            num_rows = i + 1
            with open(input_file, 'r') as f:
                first_line = f.readline()
                num_cols = len(first_line.split(','))
            array_shape = (num_rows, num_cols)
            print(f"Estimated array shape from CSV: {array_shape}")
    else:
        raise ValueError("Input file must be .npy or .csv")
    
    if output_file is not None:
        if os.path.exists(output_file):
            os.remove(output_file)
        output = np.lib.format.open_memmap(
            output_file, mode='w+', dtype=dtype, shape=array_shape
        )
    else:
        output = np.zeros(array_shape, dtype=dtype)
    
    total_rows = array_shape[0]
    
    if apply_filter:
        nyquist = 0.5
        cutoff = 0.4
        order = 3
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    for start_row in tqdm(range(0, total_rows, row_chunk), desc="Processing rows"):
        end_row = min(start_row + row_chunk, total_rows)
        
        if input_file.endswith('.npy'):
            input_mmap = np.lib.format.open_memmap(input_file, mode='r')
            chunk = np.array(input_mmap[start_row:end_row], dtype=dtype)
            del input_mmap
        else:
            chunk = pd.read_csv(
                input_file, 
                skiprows=range(1, start_row+1) if start_row > 0 else None,
                nrows=end_row-start_row
            ).values.astype(dtype)
        
        if apply_filter:
            for i in range(chunk.shape[0]):
                chunk[i, :] = filtfilt(b, a, chunk[i, :])
        
        chunk = np.clip(chunk, min_val, max_val)
        max_abs = np.max(np.abs(chunk))
        if max_abs > 0:
            chunk = chunk / max_abs
        
        output[start_row:end_row] = chunk
        
        del chunk
        gc.collect()
    
    if output_file is None:
        return output
    else:
        del output
        gc.collect()
        print(f"Processed data saved to {output_file}")
        return None

def process_existing_array(numpy_array, output_file=None, apply_filter=True, min_val=-800000, max_val=800000):
    temp_file = "temp_large_array.npy"
    np.save(temp_file, numpy_array)
    del numpy_array
    gc.collect()
    
    result = process_large_2d_array(temp_file, output_file, apply_filter=apply_filter, min_val=min_val, max_val=max_val)
    
    os.remove(temp_file)
    
    return result

def filter_regions(predicted_regions, signal_data, min_threshold=0.2):
    filtered_regions = []
    for start, end in predicted_regions:
        if start == 0:
            continue
        peak_value = np.max(signal_data[start:end])
        if peak_value < min_threshold:
            continue
        filtered_regions.append((start, end))
    return filtered_regions

def predict_chunk(model, chunk_data, device):
    """Predict on a single 1024-point chunk"""
    model.eval()
    with torch.no_grad():
        chunk_tensor = torch.FloatTensor(chunk_data).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(chunk_tensor)
        return pred.squeeze().cpu().numpy()

def process_row_in_chunks(model, row_data, device, chunk_size=1024, overlap=0.5):
    """Process a single row of data in chunks"""
    original_length = len(row_data)
    stride = int(chunk_size * (1 - overlap))
    
    # Pad data if necessary
    if len(row_data) % stride != 0:
        pad_size = stride - (len(row_data) % stride)
        row_data = np.pad(row_data, (0, pad_size), 'constant')
    
    # Initialize arrays for predictions
    full_pred = np.zeros_like(row_data)
    counts = np.zeros_like(row_data)
    
    # Process each chunk
    for start in range(0, len(row_data) - chunk_size + 1, stride):
        end = start + chunk_size
        chunk = row_data[start:end]
        
        # Get prediction for this chunk
        chunk_pred = predict_chunk(model, chunk, device)
        
        # Add to full prediction
        full_pred[start:end] += chunk_pred
        counts[start:end] += 1
    
    # Average overlapping predictions
    full_pred = np.divide(full_pred, counts, where=counts>0)
    
    # Return only the predictions for the original length
    return full_pred[:original_length]

def mask_to_region_pairs(mask, threshold=0.5):
    """Convert binary mask to region pairs (start, end) of contiguous 1s"""
    binary_mask = (mask > threshold).astype(np.int32)
    region_pairs = []
    in_region = False
    start_idx = 0
    
    for i in range(len(binary_mask)):
        if binary_mask[i] == 1 and not in_region:
            in_region = True
            start_idx = i
        elif (binary_mask[i] == 0 or i == len(binary_mask) - 1) and in_region:
            in_region = False
            end_idx = i if binary_mask[i] == 0 else i + 1
            region_pairs.append([start_idx, end_idx])
    
    return region_pairs

def predict_and_visualize_with_raw_predictions(model, raw_data, num_samples=20, device=None, best_threshold=0.5):
    if device is None:
        device = torch.device("cpu")

    # Make sure model is on CPU for inference
    model = model.to(device)
    model.eval()

    total_samples = len(raw_data)
    selected_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    selected_raw_data = raw_data[selected_indices]

    # Process data
    apply_filter = True
    filtered_data = process_existing_array(
        selected_raw_data, 
        apply_filter=apply_filter, 
        min_val=-800000, 
        max_val=800000
    )

    valid_samples = []
    valid_indices = []
    for i, sample_idx in enumerate(selected_indices):
        if np.max(np.abs(filtered_data[i])) > 0.1:
            valid_samples.append(i)
            valid_indices.append(sample_idx)

    if not valid_samples:
        print("No samples found with filtered values > 0.1")
        return []

    valid_raw_data = selected_raw_data[valid_samples]
    valid_filtered_data = filtered_data[valid_samples]

    # Process each row in chunks
    sample_pred_masks = []
    sample_pred_regions = []
    
    for i in tqdm(range(len(valid_filtered_data)), desc="Processing rows"):
        # Process this row in chunks
        row_pred = process_row_in_chunks(
            model, 
            valid_filtered_data[i], 
            device, 
            chunk_size=1024, 
            overlap=0.5
        )
        
        # Verify lengths match
        assert len(row_pred) == len(valid_filtered_data[i]), \
            f"Prediction length {len(row_pred)} doesn't match input length {len(valid_filtered_data[i])}"
        
        # Convert predictions to regions
        pred_regions = mask_to_region_pairs(row_pred, threshold=best_threshold)
        
        sample_pred_masks.append(row_pred)
        sample_pred_regions.append(pred_regions)

    figures = []
    for i, sample_idx in enumerate(valid_indices):
        filtered_pred_regions = filter_regions(
            sample_pred_regions[i], 
            valid_filtered_data[i], 
            min_threshold=0.2
        )

        if len(filtered_pred_regions) > 0:
            # Create visualization
            fig, axs = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.3})
            fig.suptitle(f"Sample {sample_idx}: Raw and Processed Data with Predictions", fontsize=16)

            x_vals = np.arange(len(valid_raw_data[i]))

            # Plot raw data
            axs[0].plot(x_vals, valid_raw_data[i], color='blue', label='Raw Signal')
            axs[0].set_title(f"Raw Data - Sample {sample_idx}")
            axs[0].set_xlabel("Time")
            axs[0].set_ylabel("Amplitude")
            axs[0].legend()

            # Plot processed data with predictions
            axs[1].plot(x_vals, valid_filtered_data[i], color='blue', label='Processed Signal')
            axs[1].plot(x_vals, sample_pred_masks[i], color='green', alpha=0.5, linewidth=1, label='Prediction Score')

            # Add prediction regions
            for start, end in filtered_pred_regions:
                rect = Rectangle(
                    (start, axs[1].get_ylim()[0]),
                    width=(end - start),
                    height=axs[1].get_ylim()[1] - axs[1].get_ylim()[0],
                    facecolor='green',
                    alpha=0.3
                )
                axs[1].add_patch(rect)

            axs[1].set_title(f"Processed Data with Predicted Regions - Sample {sample_idx}")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("Amplitude")
            axs[1].legend()

            # Plot raw data with regions
            axs[2].plot(x_vals, valid_raw_data[i], color='blue', label='Raw Signal')
            for start, end in filtered_pred_regions:
                rect = Rectangle(
                    (start, axs[2].get_ylim()[0]),
                    width=(end - start),
                    height=axs[2].get_ylim()[1] - axs[2].get_ylim()[0],
                    facecolor='green',
                    alpha=0.3
                )
                axs[2].add_patch(rect)

            axs[2].set_title(f"Raw Data with Predicted Regions - Sample {sample_idx}")
            axs[2].set_xlabel("Time")
            axs[2].set_ylabel("Amplitude")
            axs[2].legend()

            plt.tight_layout()
            figures.append(fig)

            print(f"\nSample {sample_idx}:")
            print(f"  Original predicted regions: {len(sample_pred_regions[i])}")
            print(f"  Filtered predicted regions: {len(filtered_pred_regions)}")
            print("  Regions (start, end, peak):")
            for start, end in filtered_pred_regions:
                peak = np.max(valid_filtered_data[i][start:end])
                print(f"    {start} - {end} (peak: {peak:.4f})")

    return figures

def save_prediction_results(figures, output_dir='predictions'):
    """Save prediction results to files"""
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Save each figure
    for i, fig in enumerate(figures):
        filename = os.path.join(output_dir, f'prediction_{timestamp}_{i}.png')
        fig.savefig(filename)
        print(f"Saved figure to {filename}")

if __name__ == "__main__":
    print(f"Starting prediction process at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: naimur978")

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Load test data
    test_dataa = np.load('test_dataa.npy')

    # Load model weights for CPU inference
    model = UNet1D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    device = torch.device("cpu")

    # Set best threshold
    best_threshold = 0.5

    # Predict and visualize
    figures = predict_and_visualize_with_raw_predictions(
        model, 
        test_dataa, 
        num_samples=7, 
        device=device, 
        best_threshold=best_threshold
    )

    if figures:
        print(f"\nFound {len(figures)} samples with predicted regions")
        
        # Save figures to files
        save_prediction_results(figures)
        
        # Display figures
        for i, fig in enumerate(figures):
            plt.figure(fig.number)
            plt.show()
    else:
        print("\nNo samples with both sufficient signal strength and predicted regions found")

    print(f"\nPrediction process completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
