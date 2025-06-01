from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, HTMLResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import io
import base64
from pd_inference import (
    process_existing_array, 
    UNet1D, 
    process_row_in_chunks, 
    mask_to_region_pairs, 
    filter_regions
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the model at startup
device = torch.device("cpu")
model = UNet1D()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to PD Inference API"}

@app.get("/test")
async def test_visualization():
    try:
        # Load test data
        test_data = np.load("test_dataa.npy", mmap_mode='r')  # Memory-mapped mode for large files
        
        # Select samples (using first 7 samples like in your script)
        num_samples = 7
        selected_raw_data = test_data[:num_samples].copy()  # Create a copy to avoid mmap issues
        
        # Process data
        filtered_data = process_existing_array(
            selected_raw_data, 
            apply_filter=True, 
            min_val=-800000, 
            max_val=800000
        )

        # Find valid samples (those with sufficient signal strength)
        valid_samples = []
        valid_indices = []
        for i in range(len(filtered_data)):
            if np.max(np.abs(filtered_data[i])) > 0.1:
                valid_samples.append(i)
                valid_indices.append(i)

        if not valid_samples:
            raise HTTPException(status_code=404, detail="No samples found with filtered values > 0.1")

        valid_raw_data = selected_raw_data[valid_samples]
        valid_filtered_data = filtered_data[valid_samples]

        # Process each row in chunks
        sample_pred_masks = []
        sample_pred_regions = []
        
        for i in range(len(valid_filtered_data)):
            # Process this row in chunks
            row_pred = process_row_in_chunks(
                model, 
                valid_filtered_data[i], 
                device, 
                chunk_size=1024, 
                overlap=0.5
            )
            
            # Convert predictions to regions
            pred_regions = mask_to_region_pairs(row_pred, threshold=0.5)
            
            sample_pred_masks.append(row_pred)
            sample_pred_regions.append(pred_regions)

        # Create HTML content to display all plots
        html_content = """
        <html>
            <head>
                <title>PD Signal Analysis - All Samples</title>
                <style>
                    .plot-container {
                        margin: 20px 0;
                        text-align: center;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                    }
                </style>
            </head>
            <body style="margin: 20px;">
        """

        for i, sample_idx in enumerate(valid_indices):
            filtered_pred_regions = filter_regions(
                sample_pred_regions[i], 
                valid_filtered_data[i], 
                min_threshold=0.2
            )

            if len(filtered_pred_regions) > 0:
                # Create visualization with optimized settings
                plt.ioff()  # Turn off interactive mode
                fig, axs = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={'hspace': 0.3}, dpi=100)
                
                x_vals = np.arange(len(valid_raw_data[i]))
                x_vals_down = x_vals[::10]  # Downsample for plotting
                
                # Plot raw data (downsampled)
                axs[0].plot(x_vals_down, valid_raw_data[i][::10], color='blue', label='Raw Signal')
                axs[0].set_title(f"Raw Data - Sample {sample_idx}")
                
                # Plot processed data with predictions (downsampled)
                axs[1].plot(x_vals_down, valid_filtered_data[i][::10], color='blue', label='Processed')
                axs[1].plot(x_vals_down, sample_pred_masks[i][::10], color='green', alpha=0.5, label='Prediction')
                
                # Add prediction regions
                for start, end in filtered_pred_regions:
                    for ax in [axs[1], axs[2]]:
                        ax.axvspan(start, end, color='green', alpha=0.2)
                
                # Plot raw data with regions (downsampled)
                axs[2].plot(x_vals_down, valid_raw_data[i][::10], color='blue', label='Raw Signal')
                
                # Set common properties
                for ax in axs:
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Amplitude")
                    ax.legend(loc='upper right', fontsize='small')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot without the optimize parameter
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, pad_inches=0.1)
                buf.seek(0)
                plt.close(fig)

                # Add image to HTML
                img_base64 = base64.b64encode(buf.getvalue()).decode()
                html_content += f"""
                <div class="plot-container">
                    <img src="data:image/png;base64,{img_base64}" />
                    <p>
                        Sample {sample_idx}:<br>
                        Original predicted regions: {len(sample_pred_regions[i])}<br>
                        Filtered predicted regions: {len(filtered_pred_regions)}<br>
                        Regions (start, end, peak):<br>
                    </p>
                    <pre>
                """
                
                for start, end in filtered_pred_regions:
                    peak = np.max(valid_filtered_data[i][start:end])
                    html_content += f"        {start} - {end} (peak: {peak:.4f})\n"
                
                html_content += "    </pre></div>"

        html_content += """
            </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        import traceback
        error_message = f"Error during visualization: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-text")
async def test_text_only():
    try:
        print("Starting test-text analysis")
        
        # Load test data
        print("Loading test data...")
        test_data = np.load("test_dataa.npy", mmap_mode='r')
        
        # Select samples (using first 7 samples)
        num_samples = 7
        print(f"Selecting {num_samples} samples...")
        selected_raw_data = test_data[:num_samples].copy()
        print(f"Selected data shape: {selected_raw_data.shape}")
        
        # Process data
        print("Processing data...")
        filtered_data = process_existing_array(
            selected_raw_data, 
            apply_filter=True, 
            min_val=-800000, 
            max_val=800000
        )
        print("Data processing complete")

        # Find valid samples
        print("Finding valid samples...")
        valid_samples = []
        for i in range(len(filtered_data)):
            if np.max(np.abs(filtered_data[i])) > 0.1:
                valid_samples.append(i)
        print(f"Found {len(valid_samples)} valid samples")

        if not valid_samples:
            raise HTTPException(status_code=404, detail="No samples found with filtered values > 0.1")

        valid_raw_data = selected_raw_data[valid_samples]
        valid_filtered_data = filtered_data[valid_samples]

        # Process each row in chunks and collect results
        print("Processing rows in chunks...")
        results = []
        for i in range(len(valid_filtered_data)):
            print(f"Processing row {i+1}/{len(valid_filtered_data)}...")
            row_pred = process_row_in_chunks(
                model, 
                valid_filtered_data[i], 
                device, 
                chunk_size=1024, 
                overlap=0.5
            )
            print(f"Generated predictions for row {i+1}")
            
            # Convert predictions to regions
            pred_regions = mask_to_region_pairs(row_pred, threshold=0.5)
            print(f"Found {len(pred_regions)} regions in row {i+1}")
            
            # Add sample results
            sample_result = {
                "sample_index": valid_samples[i],
                "num_regions_detected": len(pred_regions),
                "regions": [
                    {
                        "start": int(start),
                        "end": int(end),
                        "duration": int(end - start)
                    }
                    for start, end in pred_regions
                ],
                "signal_stats": {
                    "max_amplitude": float(np.max(np.abs(valid_filtered_data[i]))),
                    "mean_amplitude": float(np.mean(np.abs(valid_filtered_data[i]))),
                    "std_amplitude": float(np.std(valid_filtered_data[i]))
                }
            }
            results.append(sample_result)
            print(f"Completed processing row {i+1}")

        response_data = {
            "total_samples_processed": num_samples,
            "valid_samples_found": len(valid_samples),
            "results": results
        }
        print("Analysis complete, returning results")
        return response_data
        
    except Exception as e:
        import traceback
        error_message = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/healthz")
def health_check():
    """Simple health check endpoint that should respond immediately"""
    return {"status": "ok", "model_loaded": model is not None}

