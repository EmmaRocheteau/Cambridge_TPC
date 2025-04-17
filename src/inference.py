import argparse
import torch
import time
import numpy as np
import yaml
from pathlib import Path


def load_optimized_model(model_path: str):
    """Load an optimized JIT-traced model."""
    return torch.jit.load(model_path)


def run_inference_benchmark(model, inputs, num_runs=100):
    """Run inference benchmark on the model."""
    x, flat = inputs

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(x, flat, torch.tensor(0))

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(x, flat, torch.tensor(0))
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Run inference with an optimized model")
    parser.add_argument("--model", type=str, required=True, help="Path to optimized model file (.pt)")
    parser.add_argument("--inputs", type=str, help="Path to example inputs file (optional)")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of runs for benchmark")
    args = parser.parse_args()

    # Load the model
    model = load_optimized_model(args.model)
    print(f"Loaded optimized model from {args.model}")

    # Load or generate inputs
    if args.inputs and Path(args.inputs).exists():
        inputs = torch.load(args.inputs)
        x, flat = inputs
    else:
        # Load model info to determine input shapes
        model_info_path = Path(args.model).parent / "model_info.yaml"
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = yaml.safe_load(f)

            x_shape = model_info.get("input_shape_x", [1, 48, 257])
            flat_shape = model_info.get("input_shape_flat", [1, 9])

            x = torch.zeros(*x_shape)
            flat = torch.zeros(*flat_shape)
        else:
            # Default shapes if no info available
            print("Warning: No model info found, using default input shapes")
            x = torch.zeros(1, 48, 257)
            flat = torch.zeros(1, 9)

    print(f"Input shapes: x={x.shape}, flat={flat.shape}")

    # Run a single inference
    with torch.no_grad():
        output = model(x, flat, torch.tensor(0))

    print(f"Output shape: {output.shape}")

    # Run benchmark if requested
    if args.benchmark:
        print(f"Running inference benchmark with {args.num_runs} iterations...")
        avg_time = run_inference_benchmark(model, (x, flat), args.num_runs)
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {1 / avg_time:.2f} inferences/second")


if __name__ == "__main__":
    main()
