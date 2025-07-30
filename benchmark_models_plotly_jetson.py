import argparse
import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime as ort
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
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

class ModelBenchmarkPlotly:
    def __init__(self, onnx_path, tensorrt_path=None, test_images_dir=None, num_iterations=1000):
        self.onnx_path = onnx_path
        self.tensorrt_path = tensorrt_path
        self.test_images_dir = test_images_dir
        self.num_iterations = num_iterations

        self.onnx_times = deque(maxlen=100)
        self.tensorrt_times = deque(maxlen=100)
        self.onnx_fps = deque(maxlen=100)
        self.tensorrt_fps = deque(maxlen=100)
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

        self.init_models()
        self.test_images = self.load_test_images()

    def init_models(self):
        print("Initializing models...")
        self.onnx_session = ort.InferenceSession(self.onnx_path)
        print("ONNX model loaded successfully")
        if TENSORRT_AVAILABLE and self.tensorrt_path and os.path.exists(self.tensorrt_path):
            self.init_tensorrt()
        else:
            self.tensorrt_available = False
            print("TensorRT not available or engine file not found")

    def init_tensorrt(self):
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.tensorrt_path, 'rb') as f:
                engine_data = f.read()
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            input_shape = (1, 3, 224, 224)
            output_shape = (1, 4)
            #self.d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
            #self.d_output = cuda.mem_alloc(np.prod(output_shape) * 4)
            self.d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))
            self.d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))

            self.tensorrt_available = True
            print("TensorRT engine loaded successfully")
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            self.tensorrt_available = False

   
        
    def load_test_images(self):
        test_images = []
        if self.test_images_dir and os.path.exists(self.test_images_dir):
            for root, _, files in os.walk(self.test_images_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            img_path = os.path.join(root, file)
                            img = Image.open(img_path).convert('RGB')
                            test_images.append(img)
                        except Exception as e:
                            print(f"Failed to load image {file}: {e}")
        if not test_images:
            print("No test images found, creating dummy images...")
            for i in range(10):
                test_images.append(Image.new('RGB', (224, 224), color=(i * 25, i * 25, i * 25)))
        return test_images


    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).numpy()

    def run_onnx_inference(self, input_tensor):
        start_time = time.time()
        outputs = self.onnx_session.run(None, {'input': input_tensor})
        inference_time = (time.time() - start_time) * 1000
        predictions = np.argmax(outputs[0], axis=1)
        return inference_time, predictions[0]

    def run_tensorrt_inference(self, input_tensor):
        if not self.tensorrt_available:
            return None, None
        self.context.set_input_shape("input", input_tensor.shape)
        start_time = time.time()
        cuda.memcpy_htod(self.d_input, input_tensor.astype(np.float32))
        self.context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])
        output = np.empty((1, 4), dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)
        inference_time = (time.time() - start_time) * 1000
        predictions = np.argmax(output, axis=1)
        return inference_time, predictions[0]

    def benchmark_models(self):
        print("Starting benchmark...")
        for i in range(self.num_iterations):
            img = self.test_images[np.random.randint(len(self.test_images))]
            input_tensor = self.preprocess_image(img)
            onnx_time, _ = self.run_onnx_inference(input_tensor)
            self.onnx_times.append(onnx_time)
            self.onnx_fps.append(1000 / onnx_time if onnx_time > 0 else 0)
            if self.tensorrt_available:
                tensorrt_time, _ = self.run_tensorrt_inference(input_tensor)
                if tensorrt_time is not None:
                    self.tensorrt_times.append(tensorrt_time)
                    self.tensorrt_fps.append(1000 / tensorrt_time if tensorrt_time > 0 else 0)
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{self.num_iterations}")

    def create_benchmark_plots(self):
        """Create comprehensive benchmark plots using Plotly."""
        # Convert deques to lists for plotting
        onnx_times_list = list(self.onnx_times)
        tensorrt_times_list = list(self.tensorrt_times) if self.tensorrt_available else []
        onnx_fps_list = list(self.onnx_fps)
        tensorrt_fps_list = list(self.tensorrt_fps) if self.tensorrt_available else []
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Inference Time Distribution',
                'FPS Over Time',
                'Average Inference Time',
                'Inference Time Trend',
                'Performance Summary',
                'Speedup Comparison'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "table"}, {"type": "bar"}]
            ]
        )
        
        # 1. Inference Time Distribution
        fig.add_trace(
            go.Histogram(
                x=onnx_times_list,
                name='ONNX',
                marker_color='#2E86AB',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=1
        )
        
        if tensorrt_times_list:
            fig.add_trace(
                go.Histogram(
                    x=tensorrt_times_list,
                    name='TensorRT',
                    marker_color='#A23B72',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
        
        # 2. FPS Over Time
        x_axis = list(range(len(onnx_fps_list)))
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=onnx_fps_list,
                name='ONNX',
                line=dict(color='#2E86AB', width=2),
                mode='lines'
            ),
            row=1, col=2
        )
        
        if tensorrt_fps_list:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=tensorrt_fps_list,
                    name='TensorRT',
                    line=dict(color='#A23B72', width=2),
                    mode='lines'
                ),
                row=1, col=2
            )
        
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
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=times,
                name='Average Time',
                marker_color=colors,
                text=[f'{t:.2f}ms' for t in times],
                textposition='auto',
                textfont=dict(size=14, color='white')
            ),
            row=2, col=1
        )
        
        # 4. Inference Time Trend (Moving Average)
        window_size = 50
        if len(onnx_times_list) >= window_size:
            onnx_moving_avg = [np.mean(onnx_times_list[i:i+window_size]) 
                              for i in range(len(onnx_times_list) - window_size + 1)]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(onnx_moving_avg))),
                    y=onnx_moving_avg,
                    name='ONNX Trend',
                    line=dict(color='#2E86AB', width=2),
                    mode='lines'
                ),
                row=2, col=2
            )
        
        if self.tensorrt_available and len(tensorrt_times_list) >= window_size:
            tensorrt_moving_avg = [np.mean(tensorrt_times_list[i:i+window_size]) 
                                  for i in range(len(tensorrt_times_list) - window_size + 1)]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(tensorrt_moving_avg))),
                    y=tensorrt_moving_avg,
                    name='TensorRT Trend',
                    line=dict(color='#A23B72', width=2),
                    mode='lines'
                ),
                row=2, col=2
            )
        
        # 5. Performance Summary Table
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
        
        # Add table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'ONNX'] + (['TensorRT'] if tensorrt_stats else []),
                    fill_color='#2E86AB',
                    font=dict(color='white', size=14),
                    align='center'
                ),
                cells=dict(
                    values=[[row[i] for row in table_data] for i in range(len(table_data[0]))],
                    fill_color=[['#F8F9FA', '#E9ECEF'] * (len(table_data) // 2 + 1)][:len(table_data)],
                    font=dict(size=12),
                    align='center'
                )
            ),
            row=3, col=1
        )
        
        # 6. Speedup Comparison
        if self.tensorrt_available and tensorrt_times_list:
            speedup = onnx_stats['Mean Time (ms)'] / tensorrt_stats['Mean Time (ms)']
            efficiency = (tensorrt_stats['Mean FPS'] / onnx_stats['Mean FPS']) * 100
            
            fig.add_trace(
                go.Bar(
                    x=['Speedup', 'Efficiency (%)'],
                    y=[speedup, efficiency],
                    name='Performance Gain',
                    marker_color=['#28A745', '#FFC107'],
                    text=[f'{speedup:.2f}x', f'{efficiency:.1f}%'],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                ),
                row=3, col=2
            )
        else:
            fig.add_annotation(
                text="TensorRT not available<br>for comparison",
                xref="x3", yref="y3",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Model Performance Benchmark Comparison',
                x=0.5,
                font=dict(size=24, color='black')
            ),
            height=1200,
            width=1600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=12)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Inference Time (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Inference Number", row=1, col=2)
        fig.update_yaxes(title_text="FPS", row=1, col=2)
        fig.update_yaxes(title_text="Average Time (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Window Position", row=2, col=2)
        fig.update_yaxes(title_text="Moving Average Time (ms)", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=3, col=2)
        
        return fig
        
    def run_benchmark(self):
        print("Starting benchmark test...")
        self.benchmark_models()
        print("Creating performance plots...")
        fig = self.create_benchmark_plots()
        print("Displaying results...")
        fig.show()

        # Save plot as PNG
        png_path = "/workspace/results/benchmark_plot.png"
        try:
            fig.write_image(png_path)
            print(f"✅ Plot saved as PNG to {png_path}")
        except Exception as e:
            print(f"⚠️ Failed to save PNG: {e}")

        self.print_summary()

    def print_summary(self):
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"ONNX Performance:")
        print(f"  - Mean inference time: {np.mean(self.onnx_times):.2f} ms")
        print(f"  - Std deviation: {np.std(self.onnx_times):.2f} ms")
        print(f"  - Mean FPS: {np.mean(self.onnx_fps):.2f}")
        print(f"  - Max FPS: {np.max(self.onnx_fps):.2f}")
        if self.tensorrt_available and self.tensorrt_times:
            print(f"\nTensorRT Performance:")
            print(f"  - Mean inference time: {np.mean(self.tensorrt_times):.2f} ms")
            print(f"  - Std deviation: {np.std(self.tensorrt_times):.2f} ms")
            print(f"  - Mean FPS: {np.mean(self.tensorrt_fps):.2f}")
            print(f"  - Max FPS: {np.max(self.tensorrt_fps):.2f}")
            speedup = np.mean(self.onnx_times) / np.mean(self.tensorrt_times)
            print(f"\nPerformance Comparison:")
            print(f"  - TensorRT speedup: {speedup:.2f}x")
            print(f"  - Efficiency improvement: {(speedup - 1) * 100:.1f}%")
        print("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark ONNX and TensorRT models with Plotly.')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--tensorrt_path', type=str, help='Path to TensorRT engine')
    parser.add_argument('--test_images_dir', type=str, help='Directory with test images')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of benchmark iterations')
    return parser.parse_args()

def main():
    args = parse_args()
    benchmark = ModelBenchmarkPlotly(
        onnx_path=args.onnx_path,
        tensorrt_path=args.tensorrt_path,
        test_images_dir=args.test_images_dir,
        num_iterations=args.num_iterations
    )
    benchmark.run_benchmark()

if __name__ == '__main__':
    main()

