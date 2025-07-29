# Model Performance Benchmarking Script

This script provides comprehensive benchmarking capabilities for comparing ONNX and TensorRT model performance with detailed visualizations and metrics.

## Features

### 📊 **Comprehensive Performance Metrics**
- **Inference Time Distribution**: Histogram showing the distribution of inference times
- **FPS Over Time**: Real-time FPS tracking during benchmark runs
- **Average Time Breakdown**: Bar chart comparing mean inference times
- **Inference Time Trend**: Moving average analysis showing performance stability
- **Performance Summary Table**: Detailed statistics including mean, std, min, max times and FPS
- **Speedup Comparison**: Direct comparison between ONNX and TensorRT performance

### 🎯 **Key Capabilities**
- **Dual Model Support**: Benchmarks both ONNX and TensorRT models simultaneously
- **Real-time Monitoring**: Tracks performance metrics during benchmark execution
- **Flexible Input**: Supports custom test image directories or generates dummy images
- **Configurable Iterations**: Adjustable number of benchmark iterations
- **GPU Memory Management**: Proper CUDA memory allocation for TensorRT inference
- **Error Handling**: Graceful handling of missing TensorRT or model files

### 📈 **Visualization Components**

#### 1. **Inference Time Distribution**
- Histogram showing frequency distribution of inference times
- Separate overlays for ONNX and TensorRT models
- Helps identify performance consistency and outliers

#### 2. **FPS Over Time**
- Line plot showing FPS variation during benchmark
- Real-time performance tracking
- Identifies performance degradation or improvement patterns

#### 3. **Average Time Breakdown**
- Bar chart comparing mean inference times
- Value labels on bars for precise readings
- Clear visual comparison between models

#### 4. **Inference Time Trend**
- Moving average analysis with configurable window size
- Shows performance stability over time
- Helps identify warm-up effects or performance drift

#### 5. **Performance Summary Table**
- Comprehensive statistics table
- Includes mean, standard deviation, min, max times and FPS
- Side-by-side comparison of both models

#### 6. **Speedup Comparison**
- Direct performance comparison metrics
- Speedup multiplier and efficiency percentage
- Visual representation of performance gains

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for TensorRT)
- NVIDIA drivers and CUDA toolkit

### Install Dependencies
```bash
pip install -r requirements_benchmark.txt
```

### Optional TensorRT Installation
For TensorRT benchmarking, install TensorRT following NVIDIA's official guide:
```bash
# For Ubuntu/Debian
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev

# Or install from NVIDIA repository
# Follow instructions at: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/
```

## Usage

### Basic Usage (ONNX Only)
```bash
python benchmark_models.py --onnx_path path/to/model.onnx
```

### Full Benchmark (ONNX + TensorRT)
```bash
python benchmark_models.py \
    --onnx_path path/to/model.onnx \
    --tensorrt_path path/to/model.engine \
    --test_images_dir path/to/test/images \
    --num_iterations 1000
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--onnx_path` | str | Yes | Path to ONNX model file |
| `--tensorrt_path` | str | No | Path to TensorRT engine file |
| `--test_images_dir` | str | No | Directory containing test images |
| `--num_iterations` | int | No | Number of benchmark iterations (default: 1000) |

## Output

### 📊 **Graphical Output**
The script displays a comprehensive dashboard with 6 subplots:

1. **Inference Time Distribution**: Histogram comparing time distributions
2. **FPS Over Time**: Line plot showing FPS trends
3. **Average Time Breakdown**: Bar chart of mean inference times
4. **Inference Time Trend**: Moving average analysis
5. **Performance Summary Table**: Detailed statistics
6. **Speedup Comparison**: Performance improvement metrics

### 📝 **Console Output**
Detailed performance summary including:
- Mean inference times and standard deviations
- FPS statistics (mean and maximum)
- Performance comparison metrics
- Speedup and efficiency improvements

## Example Output

```
============================================================
BENCHMARK SUMMARY
============================================================
ONNX Performance:
  - Mean inference time: 15.23 ms
  - Std deviation: 2.45 ms
  - Mean FPS: 65.67
  - Max FPS: 89.12

TensorRT Performance:
  - Mean inference time: 8.91 ms
  - Std deviation: 1.23 ms
  - Mean FPS: 112.23
  - Max FPS: 145.67

Performance Comparison:
  - TensorRT speedup: 1.71x
  - Efficiency improvement: 70.9%
============================================================
```

## Performance Metrics Explained

### **Inference Time**
- **Mean**: Average time per inference
- **Std Deviation**: Consistency measure
- **Min/Max**: Performance range

### **FPS (Frames Per Second)**
- **Mean FPS**: Average throughput
- **Max FPS**: Peak performance capability

### **Speedup**
- **Multiplier**: How many times faster TensorRT is vs ONNX
- **Efficiency**: Percentage improvement in throughput

## Use Cases

### 🔬 **Research & Development**
- Compare model optimization techniques
- Evaluate hardware performance
- Analyze model efficiency

### 🚀 **Production Deployment**
- Validate deployment configurations
- Monitor performance in production
- Optimize inference pipelines

### 📊 **Performance Analysis**
- Identify bottlenecks
- Compare different model formats
- Optimize for specific hardware

## Troubleshooting

### Common Issues

#### **TensorRT Not Available**
```
Warning: TensorRT not available. Only ONNX benchmarking will be performed.
```
**Solution**: Install TensorRT following NVIDIA's official guide

#### **CUDA Memory Errors**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use smaller test images

#### **Model Loading Errors**
```
RuntimeError: Failed to load model
```
**Solution**: Verify model file paths and format compatibility

#### **Performance Issues**
- **High variance**: Check for system load or thermal throttling
- **Low FPS**: Verify GPU utilization and memory bandwidth
- **Inconsistent results**: Ensure proper warm-up runs

## Advanced Configuration

### Custom Test Images
```bash
python benchmark_models.py \
    --onnx_path model.onnx \
    --test_images_dir /path/to/real/test/images
```

### Extended Benchmarking
```bash
python benchmark_models.py \
    --onnx_path model.onnx \
    --tensorrt_path model.engine \
    --num_iterations 5000
```

### Integration with Training Pipeline
```python
from benchmark_models import ModelBenchmark

# Create benchmark instance
benchmark = ModelBenchmark(
    onnx_path='model.onnx',
    tensorrt_path='model.engine',
    num_iterations=1000
)

# Run benchmark
benchmark.run_benchmark()
```

## Performance Tips

### 🚀 **Optimization Strategies**
1. **Warm-up Runs**: Include warm-up iterations before timing
2. **Memory Management**: Proper GPU memory allocation
3. **Batch Processing**: Consider batch inference for higher throughput
4. **Hardware Monitoring**: Monitor GPU temperature and utilization

### 📈 **Best Practices**
- Run benchmarks on dedicated hardware
- Close unnecessary applications during testing
- Use consistent input sizes and formats
- Monitor system resources during execution

## Contributing

Feel free to contribute improvements:
- Add new performance metrics
- Enhance visualizations
- Optimize memory usage
- Add support for additional model formats

## License

This project is part of the Brain Cancer MultiClass classification system. 