# Jetson Xavier AGX/Orin Benchmarking Setup

This document provides instructions for running the benchmarking scripts on Jetson Xavier AGX and Orin devices using Docker containers.

## 🎯 Overview

The Jetson devices have specific requirements:
- **Python 3.8** (not 3.10)
- **ARM64 architecture** (aarch64)
- **NVIDIA TensorRT** compatibility
- **PyCUDA** with specific versions

The Docker container handles all these requirements automatically.

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the following installed on your Jetson device:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Install NVIDIA Docker runtime
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Add user to docker group (optional)
sudo usermod -aG docker $USER
```

### 2. Run the Setup Script

```bash
# Make the script executable
chmod +x run_jetson_benchmark.sh

# Run the setup
./run_jetson_benchmark.sh
```

### 3. Access the Container

```bash
# Enter the container
docker exec -it jetson-benchmark bash

# Check system information
./check_system.sh
```

## 📁 Directory Structure

```
Brain-Cancer-MultiClass/
├── Dockerfile.jetson              # Jetson-specific Dockerfile
├── docker-compose.jetson.yml      # Docker Compose configuration
├── run_jetson_benchmark.sh        # Setup script
├── models/                        # Mounted models directory
├── results/                       # Mounted results directory
└── exported_models/               # Mounted exported models directory
```

## 🔧 Container Features

### System Information
- **Base Image**: NVIDIA L4T r35.2.1
- **Python**: 3.8
- **CUDA**: 11.4
- **TensorRT**: 8.5.1.7
- **PyCUDA**: 2022.2.2
- **ONNX Runtime**: 1.15.1 (GPU)

### Pre-installed Packages
- PyTorch (CUDA-enabled)
- TensorRT Python bindings
- ONNX Runtime GPU
- Plotly (for interactive charts)
- OpenCV
- NumPy, SciPy, Matplotlib
- All benchmarking dependencies

## 📊 Usage Examples

### 1. Basic Benchmarking

```bash
# Inside the container
python3.8 benchmark_models_plotly.py \
    --onnx_path /workspace/models/model.onnx \
    --tensorrt_path /workspace/models/model.engine \
    --output_dir /workspace/results \
    --num_runs 100 \
    --batch_size 1
```

### 2. System Check

```bash
# Check Jetson system information
./check_system.sh
```

### 3. Example Script

```bash
# Run the example script
python3.8 run_benchmark_example.py
```

### 4. Help Commands

```bash
# Get help for benchmarking scripts
python3.8 benchmark_models_plotly.py --help
python3.8 benchmark_models.py --help
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in, or run:
newgrp docker
```

#### 2. NVIDIA Docker Runtime Not Found
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

#### 3. CUDA Version Mismatch
The container uses CUDA 11.4. If you have a different version:
```bash
# Check your CUDA version
nvcc --version
# Update the Dockerfile.jetson if needed
```

#### 4. Memory Issues
Jetson devices have limited RAM. If you encounter memory issues:
```bash
# Reduce batch size
python3.8 benchmark_models_plotly.py --batch_size 1 --num_runs 50
```

#### 5. PyCUDA Installation Issues
If PyCUDA fails to install:
```bash
# Inside the container, try:
python3.8 -m pip install --no-cache-dir pycuda==2022.2.2
```

### Performance Optimization

#### 1. GPU Memory Management
```bash
# Monitor GPU usage
nvidia-smi -l 1
```

#### 2. CPU Affinity
```bash
# Set CPU affinity for better performance
taskset -c 0-3 python3.8 benchmark_models_plotly.py [options]
```

#### 3. Power Management
```bash
# Set performance mode
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks   # Max clocks
```

## 📈 Performance Expectations

### Jetson Xavier AGX
- **FP32**: ~2-4 FPS
- **FP16**: ~4-8 FPS
- **INT8**: ~8-16 FPS

### Jetson Orin
- **FP32**: ~4-8 FPS
- **FP16**: ~8-16 FPS
- **INT8**: ~16-32 FPS

*Note: Actual performance depends on model complexity and input size.*

## 🔄 Container Management

### Start Container
```bash
docker-compose -f docker-compose.jetson.yml up -d
```

### Stop Container
```bash
docker-compose -f docker-compose.jetson.yml down
```

### Rebuild Container
```bash
docker-compose -f docker-compose.jetson.yml build --no-cache
```

### View Logs
```bash
docker-compose -f docker-compose.jetson.yml logs -f
```

## 📋 Environment Variables

The container uses these environment variables:
- `CUDA_HOME=/usr/local/cuda`
- `TENSORRT_ROOT=/usr/lib/aarch64-linux-gnu`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

## 🛠️ Customization

### Modify Dockerfile
Edit `Dockerfile.jetson` to:
- Change Python version
- Add custom packages
- Modify CUDA/TensorRT versions

### Modify Docker Compose
Edit `docker-compose.jetson.yml` to:
- Add volume mounts
- Change environment variables
- Modify resource limits

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `./check_system.sh` inside the container
3. Check Docker logs: `docker logs jetson-benchmark`
4. Verify NVIDIA drivers: `nvidia-smi`

## 🔗 Useful Links

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-xavier-devkit)
- [Docker Installation for Jetson](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/) 