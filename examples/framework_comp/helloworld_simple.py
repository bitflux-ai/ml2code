#!/usr/bin/env python3
"""
Ultra-simple PyTorch model generator for framework comparison.
Creates a basic fully-connected network that takes 32 f32 inputs and produces 4 f32 outputs,
matching the native Rust model for fair comparison.
NO convolutional operations to avoid any ONNX conversion issues.
"""

import torch
import torch.nn as nn
import torch.onnx

class SimpleFC(nn.Module):
    def __init__(self, input_size=32, output_size=4):
        super(SimpleFC, self).__init__()
        
        # Simple fully connected layers only - no conv, no pooling, nothing fancy
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, output_size)
        
        # Simple activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 32)
        
        # Fully connected layers with ReLU
        x = self.relu(self.fc1(x))  # (batch_size, 16)
        x = self.relu(self.fc2(x))  # (batch_size, 8)
        x = self.fc3(x)             # (batch_size, 4)
        
        return x

def main():
    # Create model
    model = SimpleFC(input_size=32, output_size=4)
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input matching our test format
    # Shape: (batch_size=1, input_size=32)
    dummy_input = torch.randn(1, 32)
    
    # Test the model
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model input shape: {dummy_input.shape}")
        print(f"Model output shape: {output.shape}")
        print(f"Sample output: {output.numpy().flatten()}")
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(f"Input: 32 values")
    print(f"FC1: 32->16 + ReLU")
    print(f"FC2: 16->8 + ReLU")
    print(f"FC3: 8->4 (output)")
    
    # Export to ONNX
    onnx_path = "helloworld_simple.onnx"
    
    torch.onnx.export(
        model,                          # model being run
        dummy_input,                    # model input (or a tuple for multiple inputs)
        onnx_path,                     # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=11,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names=['input'],          # the model's input names
        output_names=['output'],        # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\nONNX model exported to: {onnx_path}")
    print(f"Expected input shape: [batch_size, 32]")
    print(f"Expected output shape: [batch_size, 4]")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        # Print input/output info
        print("\nONNX Model Info:")
        for input_info in onnx_model.graph.input:
            print(f"  Input: {input_info.name}, shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
        for output_info in onnx_model.graph.output:
            print(f"  Output: {output_info.name}, shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")
            
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        
        # Print ONNX operations to verify no MaxPool
        print(f"\nONNX Operations:")
        for node in onnx_model.graph.node:
            print(f"  {node.op_type}")
            
    except ImportError:
        print("Install 'onnx' package to verify the exported model: pip install onnx")

if __name__ == "__main__":
    main()
