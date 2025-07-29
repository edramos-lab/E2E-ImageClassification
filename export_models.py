import argparse
import os
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description='Export trained models to ONNX and TensorRT formats.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pt model file')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='Model architecture name')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--output_dir', type=str, default='exported_models', help='Output directory for exported models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision for TensorRT')
    parser.add_argument('--int8', action='store_true', help='Use INT8 precision for TensorRT')
    
    return parser.parse_args()

def load_model(model_path, model_name, num_classes, device):
    """Load the trained PyTorch model."""
    print(f"Creating model: {model_name}")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Invalid checkpoint format.")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("\n🔥 ERROR: Model architecture and checkpoint do not match.")
        print("Did you forget to set --model_name correctly?")
        print(f"Model name you used: {model_name}")
        raise e

    model.to(device)
    model.eval()
    return model


def export_to_onnx(model, output_path, input_size, batch_size, device):
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model exported successfully!")

def export_to_tensorrt(onnx_path, output_path, fp16=False, int8=False):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ Using FP16 mode")
        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print("✅ Using INT8 mode")

        print(f"Parsing ONNX: {onnx_path}")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("❌ Failed to parse the ONNX model")

        # 🔧 Define optimization profile for dynamic input
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = (1, 3, 224, 224)

        profile.set_shape(input_name,
                          min=input_shape,
                          opt=input_shape,
                          max=(4, 3, 224, 224))  # puedes ajustar el max batch

        config.add_optimization_profile(profile)

        print("🔧 Building serialized TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("❌ Failed to build TensorRT engine!")

        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        print("✅ TensorRT engine exported successfully!")


def test_onnx_inference(onnx_path, test_image_path, class_names):
    """Test ONNX model inference."""
    print("Testing ONNX inference...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Load and preprocess test image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).numpy()
    
    # Run inference
    outputs = ort_session.run(None, {'input': input_tensor})
    predictions = np.argmax(outputs[0], axis=1)
    
    print(f"ONNX Prediction: {class_names[predictions[0]]}")
    return predictions[0]

def test_tensorrt_inference(engine_path, test_image_path, class_names):
    """Test TensorRT model inference."""
    print("Testing TensorRT inference...")
    
    # Load TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    # Allocate GPU memory
    input_shape = (1, 3, 224, 224)
    output_shape = (1, 4)
    
    d_input = cuda.mem_alloc(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4)
    d_output = cuda.mem_alloc(output_shape[0] * output_shape[1] * 4)
    
    # Load and preprocess test image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).numpy()
    
    # Copy input to GPU
    cuda.memcpy_htod(d_input, input_tensor.astype(np.float32))
    
    # Run inference
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    
    # Copy output from GPU
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    
    predictions = np.argmax(output, axis=1)
    print(f"TensorRT Prediction: {class_names[predictions[0]]}")
    return predictions[0]

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Class names (update based on your dataset)
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.model_name, args.num_classes, device)
    
    # Generate output filenames
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
    tensorrt_path = os.path.join(args.output_dir, f"{model_name}.engine")
    
    # Export to ONNX
    export_to_onnx(model, onnx_path, args.input_size, args.batch_size, device)
    
    # Export to TensorRT
    export_to_tensorrt(onnx_path, tensorrt_path, args.fp16, args.int8)
    
    # Test inference (if test image is provided)
    test_image_path = "test_image.jpg"  # Update with your test image path
    if os.path.exists(test_image_path):
        print("\nTesting model inference...")
        onnx_pred = test_onnx_inference(onnx_path, test_image_path, class_names)
        tensorrt_pred = test_tensorrt_inference(tensorrt_path, test_image_path, class_names)
        
        print(f"ONNX prediction: {class_names[onnx_pred]}")
        print(f"TensorRT prediction: {class_names[tensorrt_pred]}")
    
    print(f"\nExport completed!")
    print(f"ONNX model: {onnx_path}")
    print(f"TensorRT model: {tensorrt_path}")

if __name__ == '__main__':
    main() 