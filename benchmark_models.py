import argparse
import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import deque
import threading
import queue
import cv2

# Optional TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available. Only ONNX benchmarking will be performed.")

class ModelBenchmark:
    def __init__(self, onnx_path, tensorrt_path=None, test_images_dir=None, num_iterations=1000):
        self.onnx_path = onnx_path
        self.tensorrt_path = tensorrt_path
        self.test_images_dir = test_images_dir
        self.num_iterations = num_iterations
        
        # Performance tracking
        self.onnx_times = deque(maxlen=100)
        self.tensorrt_times = deque(maxlen=100)
        self.onnx_fps = deque(maxlen=100)
        self.tensorrt_fps = deque(maxlen=100)
        
        # Class names
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        # Initialize models
        self.init_models()
        
        # Load test images
        self.test_images = self.load_test_images()
        
    def init_models(self):
        """Initialize ONNX and TensorRT models."""
        print("Initializing models...")
        
        # Initialize ONNX
        self.onnx_session = ort.InferenceSession(self.onnx_path)
        print("ONNX model loaded successfully")
        
        # Initialize TensorRT if available
        if TENSORRT_AVAILABLE and self.tensorrt_path and os.path.exists(self.tensorrt_path):
            self.init_tensorrt()
        else:
            self.tensorrt_available = False
            print("TensorRT not available or engine file not found")
    
    def init_tensorrt(self):
        """Initialize TensorRT engine."""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.tensorrt_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Allocate GPU memory
            input_shape = (1, 3, 224, 224)
            output_shape = (1, 4)
            
            self.d_input = cuda.mem_alloc(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4)
            self.d_output = cuda.mem_alloc(output_shape[0] * output_shape[1] * 4)
            
            self.tensorrt_available = True
            print("TensorRT engine loaded successfully")
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            self.tensorrt_available = False
    
    def load_test_images(self):
        """Load test images from directory."""
        test_images = []
        if self.test_images_dir and os.path.exists(self.test_images_dir):
            for file in os.listdir(self.test_images_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.test_images_dir, file)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        test_images.append(img)
                    except Exception as e:
                        print(f"Failed to load {img_path}: {e}")
        
        # If no test images found, create dummy images
        if not test_images:
            print("No test images found, creating dummy images...")
            for i in range(10):
                dummy_img = Image.new('RGB', (224, 224), color=(i * 25, i * 25, i * 25))
                test_images.append(dummy_img)
        
        return test_images
    
    def preprocess_image(self, image):
        """Preprocess image for inference."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).numpy()
        return input_tensor
    
    def run_onnx_inference(self, input_tensor):
        """Run ONNX inference."""
        start_time = time.time()
        outputs = self.onnx_session.run(None, {'input': input_tensor})
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        predictions = np.argmax(outputs[0], axis=1)
        
        return inference_time, predictions[0]
    
    def run_tensorrt_inference(self, input_tensor):
    #"""Run TensorRT inference."""
        if not self.tensorrt_available:
            return None, None

        batch_size = input_tensor.shape[0]
        input_shape = (batch_size, 3, 224, 224)
        output_shape = (batch_size, 4)

        # Set dynamic shape before running inference
        self.context.set_input_shape("input", input_shape)  # Use set_input_shape instead

        start_time = time.time()

        # Copy input to GPU
        cuda.memcpy_htod(self.d_input, input_tensor.astype(np.float32))

        # Run inference
        self.context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])

        # Copy output from GPU
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)

        end_time = time.time()

        inference_time = (end_time - start_time) * 1000  # ms
        predictions = np.argmax(output, axis=1)

        return inference_time, predictions[0]

    
    def benchmark_models(self):
        """Run benchmark tests."""
        print("Starting benchmark...")
        
        for i in range(self.num_iterations):
            # Select random test image
            img = self.test_images[np.random.randint(len(self.test_images))]
            input_tensor = self.preprocess_image(img)
            
            # ONNX inference
            onnx_time, onnx_pred = self.run_onnx_inference(input_tensor)
            self.onnx_times.append(onnx_time)
            self.onnx_fps.append(1000 / onnx_time if onnx_time > 0 else 0)
            
            # TensorRT inference
            if self.tensorrt_available:
                tensorrt_time, tensorrt_pred = self.run_tensorrt_inference(input_tensor)
                if tensorrt_time is not None:
                    self.tensorrt_times.append(tensorrt_time)
                    self.tensorrt_fps.append(1000 / tensorrt_time if tensorrt_time > 0 else 0)
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{self.num_iterations}")
    
    def create_benchmark_plots(self):
        """Create comprehensive benchmark plots."""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 18))
        fig.suptitle('Model Performance Benchmark Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        # Set style for better readability
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Convert deques to lists for plotting
        onnx_times_list = list(self.onnx_times)
        tensorrt_times_list = list(self.tensorrt_times) if self.tensorrt_available else []
        onnx_fps_list = list(self.onnx_fps)
        tensorrt_fps_list = list(self.tensorrt_fps) if self.tensorrt_available else []
        
        # 1. Inference Time Distribution
        ax1.hist(onnx_times_list, bins=30, alpha=0.7, label='ONNX', color='#2E86AB', edgecolor='black', linewidth=0.5)
        if tensorrt_times_list:
            ax1.hist(tensorrt_times_list, bins=30, alpha=0.7, label='TensorRT', color='#A23B72', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Inference Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax1.set_title('Inference Time Distribution', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # 2. FPS Over Time
        x_axis = range(len(onnx_fps_list))
        ax2.plot(x_axis, onnx_fps_list, label='ONNX', color='#2E86AB', linewidth=2.5, alpha=0.8)
        if tensorrt_fps_list:
            ax2.plot(x_axis, tensorrt_fps_list, label='TensorRT', color='#A23B72', linewidth=2.5, alpha=0.8)
        ax2.set_xlabel('Inference Number', fontsize=14, fontweight='bold')
        ax2.set_ylabel('FPS', fontsize=14, fontweight='bold')
        ax2.set_title('FPS Over Time', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=12, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # 3. Average Time Breakdown
        onnx_avg = np.mean(onnx_times_list) if onnx_times_list else 0
        tensorrt_avg = np.mean(tensorrt_times_list) if tensorrt_times_list else 0
        
        models = ['ONNX']
        times = [onnx_avg]
        colors = ['#2E86AB']
        
        if self.tensorrt_available and tensorrt_times_list:
            models.append('TensorRT')
            times.append(tensorrt_avg)
            colors.append('#A23B72')
        
        bars = ax3.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Average Time (ms)', fontsize=14, fontweight='bold')
        ax3.set_title('Average Inference Time', fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        # Add value labels on bars with better positioning
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(times) * 0.02,
                    f'{time_val:.2f}ms', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. Inference Time Trend
        window_size = 50
        if len(onnx_times_list) >= window_size:
            onnx_moving_avg = [np.mean(onnx_times_list[i:i+window_size]) 
                              for i in range(len(onnx_times_list) - window_size + 1)]
            ax4.plot(range(len(onnx_moving_avg)), onnx_moving_avg, 
                    label='ONNX', color='#2E86AB', linewidth=2.5, alpha=0.8)
        
        if self.tensorrt_available and len(tensorrt_times_list) >= window_size:
            tensorrt_moving_avg = [np.mean(tensorrt_times_list[i:i+window_size]) 
                                  for i in range(len(tensorrt_times_list) - window_size + 1)]
            ax4.plot(range(len(tensorrt_moving_avg)), tensorrt_moving_avg, 
                    label='TensorRT', color='#A23B72', linewidth=2.5, alpha=0.8)
        
        ax4.set_xlabel('Window Position', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Moving Average Time (ms)', fontsize=14, fontweight='bold')
        ax4.set_title('Inference Time Trend (Moving Average)', fontsize=16, fontweight='bold', pad=20)
        ax4.legend(fontsize=12, framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        
        # 5. Performance Summary Table
        ax5.axis('tight')
        ax5.axis('off')
        
        # Calculate statistics
        onnx_stats = {
            'Mean Time (ms)': np.mean(onnx_times_list) if onnx_times_list else 0,
            'Std Time (ms)': np.std(onnx_times_list) if onnx_times_list else 0,
            'Min Time (ms)': np.min(onnx_times_list) if onnx_times_list else 0,
            'Max Time (ms)': np.max(onnx_times_list) if onnx_times_list else 0,
            'Mean FPS': np.mean(onnx_fps_list) if onnx_fps_list else 0,
            'Max FPS': np.max(onnx_fps_list) if onnx_fps_list else 0
        }
        
        tensorrt_stats = {}
        if self.tensorrt_available and tensorrt_times_list:
            tensorrt_stats = {
                'Mean Time (ms)': np.mean(tensorrt_times_list),
                'Std Time (ms)': np.std(tensorrt_times_list),
                'Min Time (ms)': np.min(tensorrt_times_list),
                'Max Time (ms)': np.max(tensorrt_times_list),
                'Mean FPS': np.mean(tensorrt_fps_list),
                'Max FPS': np.max(tensorrt_fps_list)
            }
        
        # Create table data
        table_data = []
        for stat in onnx_stats.keys():
            row = [stat, f"{onnx_stats[stat]:.2f}"]
            if tensorrt_stats:
                row.append(f"{tensorrt_stats[stat]:.2f}")
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'ONNX'] + (['TensorRT'] if tensorrt_stats else []),
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(table_data) + 1):  # +1 for header row
            for j in range(len(table_data[0]) if table_data else 0):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#2E86AB')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F8F9FA' if i % 2 == 0 else '#E9ECEF')
                    cell.set_text_props(weight='normal', color='black')
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
        
        # 6. Speedup Comparison
        if self.tensorrt_available and tensorrt_times_list:
            speedup = onnx_stats['Mean Time (ms)'] / tensorrt_stats['Mean Time (ms)']
            efficiency = (tensorrt_stats['Mean FPS'] / onnx_stats['Mean FPS']) * 100
            
            bars = ax6.bar(['Speedup', 'Efficiency (%)'], [speedup, efficiency], 
                          color=['#28A745', '#FFC107'], alpha=0.8, edgecolor='black', linewidth=1.5)
            ax6.set_ylabel('Value', fontsize=14, fontweight='bold')
            ax6.set_title('TensorRT vs ONNX Performance', fontsize=16, fontweight='bold', pad=20)
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='both', which='major', labelsize=12)
            
            # Add value labels with better positioning
            for bar, value in zip(bars, [speedup, efficiency]):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + max([speedup, efficiency]) * 0.02,
                        f'{value:.2f}x' if value == speedup else f'{value:.1f}%', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax6.text(0.5, 0.5, 'TensorRT not available\nfor comparison', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=14, fontweight='bold')
            ax6.set_title('Speedup Comparison', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout(pad=3.0)
        return fig
    
    def run_benchmark(self):
        """Run complete benchmark and display results."""
        print("Starting benchmark test...")
        self.benchmark_models()
        
        print("Creating performance plots...")
        fig = self.create_benchmark_plots()
        
        print("Displaying results...")
        plt.show()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        onnx_times_list = list(self.onnx_times)
        tensorrt_times_list = list(self.tensorrt_times) if self.tensorrt_available else []
        
        print(f"ONNX Performance:")
        print(f"  - Mean inference time: {np.mean(onnx_times_list):.2f} ms")
        print(f"  - Std deviation: {np.std(onnx_times_list):.2f} ms")
        print(f"  - Mean FPS: {np.mean(list(self.onnx_fps)):.2f}")
        print(f"  - Max FPS: {np.max(list(self.onnx_fps)):.2f}")
        
        if self.tensorrt_available and tensorrt_times_list:
            print(f"\nTensorRT Performance:")
            print(f"  - Mean inference time: {np.mean(tensorrt_times_list):.2f} ms")
            print(f"  - Std deviation: {np.std(tensorrt_times_list):.2f} ms")
            print(f"  - Mean FPS: {np.mean(list(self.tensorrt_fps)):.2f}")
            print(f"  - Max FPS: {np.max(list(self.tensorrt_fps)):.2f}")
            
            speedup = np.mean(onnx_times_list) / np.mean(tensorrt_times_list)
            print(f"\nPerformance Comparison:")
            print(f"  - TensorRT speedup: {speedup:.2f}x")
            print(f"  - Efficiency improvement: {(speedup - 1) * 100:.1f}%")
        
        print("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark ONNX and TensorRT models.')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--tensorrt_path', type=str, help='Path to TensorRT engine')
    parser.add_argument('--test_images_dir', type=str, help='Directory with test images')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of benchmark iterations')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create benchmark instance
    benchmark = ModelBenchmark(
        onnx_path=args.onnx_path,
        tensorrt_path=args.tensorrt_path,
        test_images_dir=args.test_images_dir,
        num_iterations=args.num_iterations
    )
    
    # Run benchmark
    benchmark.run_benchmark()

if __name__ == '__main__':
    main() 