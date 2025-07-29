#!/usr/bin/env python3
"""
Example script demonstrating how to use the benchmark_models.py script.
This script shows different ways to run the benchmarking tool.
"""

import subprocess
import sys
import os

def run_benchmark_example():
    """Run the benchmark with example parameters."""
    
    print("🚀 Brain Cancer Model Benchmarking Examples")
    print("=" * 50)
    
    # Example 1: Basic ONNX benchmark
    print("\n📊 Example 1: Basic ONNX Benchmark")
    print("-" * 30)
    
    onnx_model_path = "exported_models/swin_tiny_patch4_window7_224.onnx"
    
    if os.path.exists(onnx_model_path):
        cmd1 = [
            "python", "benchmark_models.py",
            "--onnx_path", onnx_model_path,
            "--num_iterations", "500"
        ]
        
        print(f"Command: {' '.join(cmd1)}")
        print("This will benchmark only the ONNX model with 500 iterations.")
        
        try:
            result = subprocess.run(cmd1, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Benchmark completed successfully!")
            else:
                print(f"❌ Benchmark failed: {result.stderr}")
        except FileNotFoundError:
            print("❌ benchmark_models.py not found. Make sure the script exists.")
    else:
        print(f"❌ ONNX model not found at: {onnx_model_path}")
        print("Please export your model to ONNX first using export_models.py")
    
    # Example 2: Full benchmark with TensorRT
    print("\n📊 Example 2: Full Benchmark (ONNX + TensorRT)")
    print("-" * 45)
    
    tensorrt_model_path = "exported_models/swin_tiny_patch4_window7_224.engine"
    test_images_dir = "Brain-Tumor-Classification2/Testing"
    
    if os.path.exists(onnx_model_path) and os.path.exists(tensorrt_model_path):
        cmd2 = [
            "python", "benchmark_models.py",
            "--onnx_path", onnx_model_path,
            "--tensorrt_path", tensorrt_model_path,
            "--test_images_dir", test_images_dir,
            "--num_iterations", "1000"
        ]
        
        print(f"Command: {' '.join(cmd2)}")
        print("This will benchmark both ONNX and TensorRT models with real test images.")
        
        try:
            result = subprocess.run(cmd2, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Full benchmark completed successfully!")
            else:
                print(f"❌ Full benchmark failed: {result.stderr}")
        except FileNotFoundError:
            print("❌ benchmark_models.py not found. Make sure the script exists.")
    else:
        print(f"❌ Models not found:")
        if not os.path.exists(onnx_model_path):
            print(f"  - ONNX model missing: {onnx_model_path}")
        if not os.path.exists(tensorrt_model_path):
            print(f"  - TensorRT engine missing: {tensorrt_model_path}")
        print("Please export your models first using export_models.py")
    
    # Example 3: Custom test images
    print("\n📊 Example 3: Custom Test Images")
    print("-" * 25)
    
    custom_test_dir = "Brain-Tumor-Classification2/Training"
    
    if os.path.exists(onnx_model_path) and os.path.exists(custom_test_dir):
        cmd3 = [
            "python", "benchmark_models.py",
            "--onnx_path", onnx_model_path,
            "--test_images_dir", custom_test_dir,
            "--num_iterations", "200"
        ]
        
        print(f"Command: {' '.join(cmd3)}")
        print("This will benchmark using custom test images from the training directory.")
        
        try:
            result = subprocess.run(cmd3, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Custom benchmark completed successfully!")
            else:
                print(f"❌ Custom benchmark failed: {result.stderr}")
        except FileNotFoundError:
            print("❌ benchmark_models.py not found. Make sure the script exists.")
    else:
        print(f"❌ Required files not found:")
        if not os.path.exists(onnx_model_path):
            print(f"  - ONNX model missing: {onnx_model_path}")
        if not os.path.exists(custom_test_dir):
            print(f"  - Test directory missing: {custom_test_dir}")
    
    print("\n" + "=" * 50)
    print("📋 Summary of Available Examples:")
    print("1. Basic ONNX benchmark (500 iterations)")
    print("2. Full benchmark with TensorRT (1000 iterations)")
    print("3. Custom test images benchmark (200 iterations)")
    print("\n💡 Tips:")
    print("- Make sure you have exported your models first")
    print("- Install TensorRT for full benchmarking")
    print("- Use real test images for more accurate results")
    print("- Adjust iterations based on your needs")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "torch", "torchvision", "onnxruntime", "numpy", 
        "matplotlib", "seaborn", "PIL", "cv2"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements_benchmark.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def main():
    """Main function to run the benchmark examples."""
    
    print("🧠 Brain Cancer Model Benchmarking Tool")
    print("=" * 40)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running benchmarks.")
        return
    
    # Run examples
    run_benchmark_example()

if __name__ == "__main__":
    main() 