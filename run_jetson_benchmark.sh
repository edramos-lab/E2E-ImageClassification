#!/bin/bash

# Jetson Benchmarking Script
# This script builds and runs the benchmarking container on Jetson Xavier AGX/Orin

set -e

echo "🚀 Jetson Benchmarking Setup"
echo "=============================="

# Check if running on Jetson
if [[ $(uname -m) != "aarch64" ]]; then
    echo "⚠️  Warning: This script is designed for Jetson devices (aarch64)"
    echo "   Current architecture: $(uname -m)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   For Jetson, follow: https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   For Jetson, follow: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if nvidia-docker is available
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  NVIDIA Docker runtime not detected. Installing nvidia-docker..."
    echo "   Please follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "   Or run: sudo apt-get install nvidia-docker2"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models results exported_models

# Build the Docker image
echo "🔨 Building Jetson benchmarking container..."
docker-compose -f docker-compose.jetson.yml build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Container built successfully!"
else
    echo "❌ Container build failed!"
    exit 1
fi

# Run the container
echo "🚀 Starting Jetson benchmarking container..."
echo ""
echo "Available commands inside the container:"
echo "  - python3.8 benchmark_models_plotly.py --help"
echo "  - python3.8 benchmark_models.py --help"
echo "  - ./check_system.sh"
echo "  - python3.8 run_benchmark_example.py"
echo ""
echo "Example usage:"
echo "  python3.8 benchmark_models_plotly.py \\"
echo "    --onnx_path /workspace/models/model.onnx \\"
echo "    --tensorrt_path /workspace/models/model.engine \\"
echo "    --output_dir /workspace/results \\"
echo "    --num_runs 100 \\"
echo "    --batch_size 1"
echo ""

# Start the container in interactive mode
docker-compose -f docker-compose.jetson.yml up -d

echo "✅ Container is running!"
echo "To access the container:"
echo "  docker exec -it jetson-benchmark bash"
echo ""
echo "To stop the container:"
echo "  docker-compose -f docker-compose.jetson.yml down" 