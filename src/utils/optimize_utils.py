import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def generate_example_inputs(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate example inputs for model tracing based on configuration.

    Args:
        config: Model configuration dictionary.

    Returns:
        Tuple of sample inputs (x, flat) for model tracing.
    """
    # Extract dimensions from config
    features = config["model"]["features"]
    batch_size = 1  # Use batch size of 1 for inference optimization
    seq_len = 48  # Default sequence length, adjust based on your use case
    no_flat_features = config["model"]["no_flat_features"]

    # Generate example inputs: [batch_size, seq_len, 2*features + 1]
    x_sample = torch.zeros(batch_size, seq_len, 2 * features + 1)

    # Add time dimension (first column)
    x_sample[:, :, 0] = torch.arange(seq_len).float().unsqueeze(0)

    # Generate flat features: [batch_size, no_flat_features]
    flat_sample = torch.zeros(batch_size, no_flat_features)

    return x_sample, flat_sample


def optimize_model(
        model_path: str,
        output_dir: Optional[str] = None,
        example_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> str:
    """
    Optimize a trained PyTorch model using JIT tracing for faster inference.

    Args:
        model_path: Path to the trained model checkpoint (.ckpt file).
        output_dir: Directory to save the optimized model (defaults to same directory as model_path).
        example_inputs: Example inputs for tracing. If None, will generate based on config.

    Returns:
        Path to the optimized model.
    """
    checkpoint_path = Path(model_path)

    if output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Load the checkpoint with map_location to support different device configurations
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract configuration
    config = checkpoint.get('hyper_parameters', {})
    if not config:
        # Try to find config.yaml in the same directory
        config_path = checkpoint_path.parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

    # Import here to avoid circular imports
    from src.models.lightning_module import HealthcarePredictionModule

    # Recreate model from checkpoint
    model = HealthcarePredictionModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        strict=False
    )

    # Extract the TPC model from the Lightning module
    tpc_model = model.model
    tpc_model.eval()  # Set to evaluation mode

    # Generate example inputs if not provided
    if example_inputs is None:
        example_inputs = generate_example_inputs(config)

    x_sample, flat_sample = example_inputs

    # Create traced/optimized model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            tpc_model,
            (x_sample, flat_sample, torch.tensor(0))  # Include time_before_pred=0
        )

    # Further optimize the model
    traced_model = torch.jit.optimize_for_inference(traced_model)

    # Save the optimized model
    optimized_model_path = os.path.join(output_dir, "model_optimized.pt")
    traced_model.save(optimized_model_path)

    # Save example inputs for verification/serving
    example_inputs_path = os.path.join(output_dir, "example_inputs.pt")
    torch.save(example_inputs, example_inputs_path)

    # Save metadata
    model_info = {
        "original_model": str(checkpoint_path),
        "input_shape_x": list(x_sample.shape),
        "input_shape_flat": list(flat_sample.shape)
    }

    with open(os.path.join(output_dir, "model_info.yaml"), "w") as f:
        yaml.dump(model_info, f)

    print(f"Optimized model saved to: {optimized_model_path}")
    return optimized_model_path


def verify_optimized_model(
        original_model_path: str,
        optimized_model_path: str,
        example_inputs_path: Optional[str] = None
) -> bool:
    """
    Verify that the optimized model produces the same outputs as the original model.

    Args:
        original_model_path: Path to the original model checkpoint.
        optimized_model_path: Path to the optimized model.
        example_inputs_path: Path to saved example inputs or None to generate new ones.

    Returns:
        True if verification passes, False otherwise.
    """
    # Import here to avoid circular imports
    from src.models.lightning_module import HealthcarePredictionModule

    # Load the original model
    original_model = HealthcarePredictionModule.load_from_checkpoint(
        original_model_path,
        map_location='cpu',
        strict=False
    ).model
    original_model.eval()

    # Load the optimized model
    optimized_model = torch.jit.load(optimized_model_path)

    # Load or generate example inputs
    if example_inputs_path and os.path.exists(example_inputs_path):
        example_inputs = torch.load(example_inputs_path)
        x_sample, flat_sample = example_inputs
    else:
        # Load config from checkpoint
        checkpoint = torch.load(original_model_path, map_location='cpu')
        config = checkpoint.get('hyper_parameters', {})
        if not config:
            # Try to find config.yaml in the same directory
            config_path = Path(original_model_path).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

        x_sample, flat_sample = generate_example_inputs(config)

    # Get outputs from both models
    with torch.no_grad():
        original_output = original_model(x_sample, flat_sample)
        optimized_output = optimized_model(x_sample, flat_sample, torch.tensor(0))

    # Compare outputs
    match = torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-5)

    if match:
        print("✅ Verification passed: Original and optimized models produce the same outputs")
    else:
        print("❌ Verification failed: Outputs differ between original and optimized models")
        # Calculate and print max difference
        max_diff = (original_output - optimized_output).abs().max().item()
        print(f"Maximum absolute difference: {max_diff:.8f}")

    return match
