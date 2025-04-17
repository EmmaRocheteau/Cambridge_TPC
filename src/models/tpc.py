"""
Temporal Pointwise Convolution (TPC) Model Implementation

This file implements the TPC architecture, a deep learning model designed specifically for
temporal healthcare data with both time-varying and static features. The TPC model was
originally introduced in "Temporal Pointwise Convolutional Networks for Length of Stay
Prediction in the Intensive Care Unit" (https://arxiv.org/abs/2007.09483).

Key Components:
1. TPCConfig - Configuration dataclass that defines the model hyperparameters
2. TemporalPointwiseLayer - Implementation of a single TPC layer
3. TempPointConv - The complete TPC model

The model processes patient data with the following structure:
- Temporal features: Measurements taken over time (e.g., vital signs, lab values)
- Static features: Patient information that doesn't change during the course of an admission (e.g., demographics)

Usage:
    config = TPCConfig(
        num_layers=6,
        dropout=0.05,
        temp_dropout_rate=0.02,
        kernel_size=3,
        temp_kernels=[6, 6, 6, 6, 6, 6],  # 6 temporal kernels per layer
        point_sizes=[14, 14, 14, 14, 14, 14],  # 14 units per pointwise layer
        momentum=0.1,  # Standard momentum for batch norm
        last_linear_size=16,  # Output dimension of 16
        features=128,
        no_flat_features=9
    )
    model = TempPointConv(config)
    output = model(x, flat)

Emma Rocheteau 2025
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from torch import Tensor, jit
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass(frozen=True)
class TPCConfig:
    """
    Configuration class for TPC model parameters (immutable).

    The TPC model processes temporal data with both time-varying and static features.
    This configuration holds all hyperparameters needed for the model architecture.

    Attributes:
        num_layers: Number of TPC layers in the model
        dropout: Dropout rate for pointwise (linear) layers (0.0-1.0)
        temp_dropout_rate: Dropout rate for temporal convolution layers (0.0-1.0)
        kernel_size: Size of the temporal convolution kernel (odd number recommended)
        temp_kernels: List of temporal kernel counts for each layer
                      (length must equal num_layers)
        point_sizes: List of output sizes for pointwise layers in each layer
                     (length must equal num_layers)
        momentum: Momentum parameter for batch normalization (typically 0.1)
        last_linear_size: Output size of the final linear layer (1 for regression)
        features: Number of features (F) for temporal convolution
        no_flat_features: Number of static (flat) features

    Example:
        config = TPCConfig(
            num_layers=6,
            dropout=0.05,
            temp_dropout_rate=0.02,
            kernel_size=3,
            temp_kernels=[6, 6, 6, 6, 6, 6],
            point_sizes=[14, 14, 14, 14, 14, 14],
            momentum=0.1,
            last_linear_size=16,
            features=128,
            no_flat_features=9
        )
    """
    num_layers: int
    dropout: float
    temp_dropout_rate: float
    kernel_size: int
    temp_kernels: List[int]
    point_sizes: List[int]
    momentum: float
    last_linear_size: int
    features: int
    no_flat_features: int


class TemporalPointwiseLayer(nn.Module):
    """
    Single Temporal Pointwise Convolution layer implementing the core TPC mechanism.

    The layer processes input in two parallel streams:
    1. Temporal stream: Applies grouped convolutions to capture temporal patterns
    2. Pointwise stream: Transforms features using linear layers

    Layer Architecture:

                         ┌─────────────────┐
                         │   Input Tensor  │
                         └─────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼
    ┌───────────────────────┐            ┌───────────────────────┐
    │    Temporal Stream    │            │   Pointwise Stream    │
    │                       │            │                       │
    │  ┌─────────────────┐  │            │  ┌─────────────────┐  │
    │  │  1D Convolution │  │            │  │ Linear Transform│  │
    │  └─────────────────┘  │            │  └─────────────────┘  │
    │           │           │            │           │           │
    │           ▼           │            │           ▼           │
    │  ┌─────────────────┐  │            │  ┌─────────────────┐  │
    │  │   Batch Norm    │  │            │  │   Batch Norm    │  │
    │  └─────────────────┘  │            │  └─────────────────┘  │
    │           │           │            │           │           │
    │           ▼           │            │           ▼           │
    │  ┌─────────────────┐  │            │  ┌─────────────────┐  │
    │  │     Dropout     │  │            │  │     Dropout     │  │
    │  └─────────────────┘  │            │  └─────────────────┘  │
    └───────────────────────┘            └───────────────────────┘
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ Skip Connections │
                         └──────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │       ReLU       │
                         └──────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │   Output Tensor  │
                         └──────────────────┘

    Key Variables and Dimensions:
    - F: Number of original features
    - Zt: Cumulative number of features added by previous pointwise layers
    - Y: Number of temporal kernels
    - B: Batch size
    - T: Sequence length
    """
    def __init__(
            self,
            in_channels: int,  # Number of input channels (F + Zt) * (Y + 1)
            temp_kernels: int,  # Number of temporal kernels (Y)
            point_size: Tuple[int, int],  # (input_size, output_size) for pointwise layer
            kernel_size: int,  # Size of temporal convolution kernel
            dilation: int,  # Dilation factor for temporal convolution
            momentum: float,  # Batch norm momentum
            dropout_rate: float,  # Dropout rate for pointwise layer
            temp_dropout_rate: float  # Dropout rate for temporal layer
    ):
        """
        Initialize a TPC layer.

        This creates both the temporal and pointwise processing streams:
        - The temporal stream uses dilated grouped convolutions
        - The pointwise stream uses linear transformations

        Parameters:
            in_channels: Number of input channels, equals (F + Zt) * (Y + 1)
                         where F is features, Zt is cumulative point features,
                         Y is temporal kernels
            temp_kernels: Number of temporal kernels (Y) for this layer
            point_size: Tuple of (input_size, output_size) for the pointwise layer
            kernel_size: Size of the temporal convolution kernel (typically 3)
            dilation: Dilation factor for temporal convolution (typically increases with depth)
            momentum: Momentum parameter for batch normalization (typically 0.1)
            dropout_rate: Dropout rate for pointwise linear layers (0.0-1.0)
            temp_dropout_rate: Dropout rate for temporal convolution (0.0-1.0)
        """
        super().__init__()

        # Temporal convolution stream
        self.temp_conv = nn.Conv1d(
            in_channels=in_channels,  # (F + Zt) * (Y + 1)
            out_channels=in_channels * temp_kernels,  # (F + Zt) * Y
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,  # Each channel is convolved separately
            padding_mode='zeros'
        )

        # BatchNorm layers
        self.bn_temp = nn.BatchNorm1d(
            in_channels * temp_kernels,
            momentum=momentum  # Using configured momentum
        )
        self.temp_dropout = nn.Dropout(p=temp_dropout_rate)

        # Pointwise transformation stream
        self.point_linear = nn.Linear(point_size[0], point_size[1])
        self.bn_point = nn.BatchNorm1d(point_size[1])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    @jit.script
    def forward(
            self,
            x: Tensor,  # Shape: [B, (F + Zt) * (Y + 1), T]
            prev_temp: Optional[Tensor],  # Shape: [(B * T), (F + Zt-1) * Y]
            prev_point: Optional[Tensor],  # Shape: [(B * T), Z]
            point_skip: Tensor,  # Shape: [B, F + Zt-1, T]
            batch_size: int,  # Batch size (B)
            seq_len: int,  # Sequence length (T)
            padding: Tuple[int, int],  # Padding for temporal convolution
            temp_kernels: int,  # Number of temporal kernels (Y)
            x_flat: Optional[Tensor] = None,  # Shape: [(B * T), no_flat_features]
            x_orig: Optional[Tensor] = None,  # Shape: [(B * T), 2F + 1]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of a TPC layer. Processes temporal and feature information in parallel.

        The process follows these steps:
        1. Temporal Stream Processing:
           - Apply padding to input tensor to maintain temporal dimension
           - Apply grouped 1D convolutions to capture temporal patterns
           - Apply batch normalization and dropout

        2. Pointwise Stream Processing:
           - Concatenate features from previous layers and static features
           - Apply linear transformation to capture feature interactions
           - Apply batch normalization and dropout

        3. Skip Connection Handling:
           - Update pointwise skip connections with previous layer outputs
           - Combine temporal and pointwise streams with skip connections
           - Apply non-linearity (ReLU) to combined representation

        Parameters:
            x: Input tensor containing temporal features [B, (F + Zt) * (Y + 1), T]
            prev_temp: Output from previous layer's temporal stream [(B * T), (F + Zt-1) * Y]
            prev_point: Output from previous layer's pointwise stream [(B * T), Z]
            point_skip: Skip connections for pointwise features [B, F + Zt-1, T]
            batch_size: Number of samples in batch (B)
            seq_len: Length of input sequences (T)
            padding: Padding sizes for temporal convolution (total_padding, 0)
            temp_kernels: Number of temporal kernels (Y)
            x_flat: Static features (if any) [(B * T), no_flat_features]
            x_orig: Original input features [(B * T), 2F + 1]

        Returns:
            temp_output: Output of temporal stream [(B * T), (F + Zt) * Y]
            point_output: Output of pointwise stream [(B * T), point_size[1]]
            next_x: Combined output for next layer [B, ((F + Zt) * (Y + 1)), T]
            point_skip: Updated skip connections [B, F + Zt, T]
        """
        # Step 1: Temporal Convolution
        # Add padding to maintain temporal dimension
        x_padded = nn.functional.pad(x, padding, 'constant', 0)  # [B, (F + Zt) * (Y + 1), T + padding]

        # Apply temporal convolution with groupwise operation
        x_temp = self.temp_dropout(
            self.bn_temp(
                self.temp_conv(x_padded)
            )
        )  # [B, (F + Zt) * Y, T]

        # Step 2: Pointwise Transformation
        # Combine all features for pointwise layer
        x_concat_parts = []
        if prev_temp is not None:
            x_concat_parts.append(prev_temp)
        if prev_point is not None:
            x_concat_parts.append(prev_point)
        if x_orig is not None:
            x_concat_parts.append(x_orig)
        if x_flat is not None:
            x_concat_parts.append(x_flat)
        x_concat = torch.cat(x_concat_parts, dim=1)  # [(B * T), sum(feature_dims)]

        # Apply pointwise transformation
        point_output = self.point_linear(x_concat)
        point_output = self.bn_point(point_output)
        point_output = self.dropout(point_output)  # [(B * T), point_size[1]]

        # Step 3: Skip Connection Handling
        # Update pointwise skip connections
        if prev_point is not None:
            point_skip = torch.cat([
                point_skip,  # [B, F + Zt-1, T]
                prev_point.view(batch_size, seq_len, -1).permute(0, 2, 1)  # [B, Z, T]
            ], dim=1)  # [B, F + Zt, T]

        # Combine temporal and pointwise streams
        temp_skip = torch.cat([
            point_skip.unsqueeze(2),  # [B, F + Zt, 1, T]
            x_temp.view(batch_size, point_skip.shape[1], temp_kernels, seq_len)  # [B, F + Zt, Y, T]
        ], dim=2)  # [B, F + Zt, Y + 1, T]

        # Prepare pointwise output for combination
        x_point_rep = point_output.view(batch_size, seq_len, -1, 1) \
            .permute(0, 2, 3, 1) \
            .repeat(1, 1, (1 + temp_kernels), 1)  # [B, point_size[1], Y + 1, T]

        # Final combination of all streams
        x_combined = torch.cat([temp_skip, x_point_rep], dim=1)
        F.relu(x_combined, inplace=True)

        # Reshape for next layer
        next_x = x_combined.contiguous().view(
            batch_size,
            (point_skip.shape[1] + point_output.shape[1]) * (1 + temp_kernels),
            seq_len
        )

        # Prepare temporal output for next layer
        temp_output = x_temp.permute(0, 2, 1).contiguous().view(
            batch_size * seq_len,
            point_skip.shape[1] * temp_kernels
        )

        return temp_output, point_output, next_x, point_skip


class TempPointConv(nn.Module):
    """
    Temporal Pointwise Convolution (TPC) model.

    This model processes temporal sequences of healthcare data by combining:
    1. Temporal convolutions to capture patterns across time
    2. Pointwise transformations to capture feature interactions
    3. Skip connections to maintain information flow

    Architecture Diagram:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                             Input                                   │
    │               [batch_size, seq_len, 2*features+1]                   │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Feature Preparation                            │
    │                               │                                     │
    │   ┌─────────────┐    ┌────────────────┐    ┌───────────────┐        │
    │   │  Features   │    │   Indicators   │    │Static Features│        │
    │   └─────────────┘    └────────────────┘    └───────────────┘        │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     TPC Layers (1...num_layers)                     │
    │                               │                                     │
    │                  (See TemporalPointwiseLayer diagram)               │
    │                               │                                     │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Final Layer                                 │
    │                               │                                     │
    │                      ┌─────────────────┐                            │
    │                      │  Linear Layer   │                            │
    │                      └─────────────────┘                            │
    │                               │                                     │
    │             [batch_size, seq_len, last_linear_size]                 │
    └─────────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, config: TPCConfig):
        """
        Initialize the TPC model.

        This method:
        1. Stores the configuration
        2. Creates caches for repeated calculations
        3. Initializes the stack of TPC layers with appropriate parameters
        4. Creates the final prediction layer
        5. Pre-calculates padding values for each layer

        Parameters:
            config: Configuration object containing model hyperparameters
                    Including number of layers, dropout rates, kernel sizes, etc.
        """
        super().__init__()
        self.config = config

        # Cache for cumulative features calculation
        self._cumulative_features_cache = {}

        # Initialize TPC layers
        self.layers = nn.ModuleList([
            TemporalPointwiseLayer(
                in_channels=self._get_layer_channels(i),
                temp_kernels=config.temp_kernels[i],
                point_size=self._get_point_size(i),
                kernel_size=config.kernel_size,
                dilation=self._get_dilation(i),
                momentum=config.momentum,
                dropout_rate=config.dropout,
                temp_dropout_rate=config.temp_dropout_rate
            )
            for i in range(config.num_layers)
        ])

        # Pre-calculate paddings for each layer
        self._paddings = [self._get_padding(layer.temp_conv) for layer in self.layers]

        # Final linear layer for predictions
        self.final_point = nn.Linear(
            in_features=self._get_final_input_size(),
            out_features=config.last_linear_size
        )

    def _get_layer_channels(self, layer_idx: int) -> int:
        """
        Calculate input channels for a given layer.

        The number of channels increases with depth due to:
        1. Skip connections from previous layers
        2. Additional features from pointwise layers

        For the first layer (layer_idx=0), this is simply 2*features (features + indicators).
        For subsequent layers, it's calculated based on the previous layer's output.

        Parameters:
            layer_idx: Index of the layer (0-based)

        Returns:
            Number of input channels for the specified layer
        """
        if layer_idx == 0:
            return 2 * self.config.features
        return (self.config.features + self._get_cumulative_features(layer_idx)) * \
            (1 + self.config.temp_kernels[layer_idx - 1])

    def _get_dilation(self, layer_idx: int) -> int:
        """
        Calculate dilation for a given layer.

        Dilation increases with depth to capture longer-range temporal dependencies.
        For layer 0, dilation is 1 (standard convolution).
        For subsequent layers, dilation increases linearly with depth.

        Parameters:
            layer_idx: Index of the layer (0-based)

        Returns:
            Dilation factor for the specified layer
        """
        return layer_idx * (self.config.kernel_size - 1) if layer_idx > 0 else 1

    def _get_cumulative_features(self, layer_idx: int) -> int:
        """
        Calculate cumulative features added by pointwise layers up to a given layer.

        This method uses caching to avoid redundant calculations. For layer 0,
        the result is always 0 as there are no previous pointwise layers.

        Parameters:
            layer_idx: Index of the layer (0-based)

        Returns:
            Sum of pointwise feature dimensions up to (but not including) the specified layer
        """
        if layer_idx not in self._cumulative_features_cache:
            self._cumulative_features_cache[layer_idx] = sum(self.config.point_sizes[:layer_idx])
        return self._cumulative_features_cache[layer_idx]

    def _get_point_size(self, layer_idx: int) -> Tuple[int, int]:
        """
        Calculate input and output sizes for pointwise layer.

        This method computes:
        1. The total size of all inputs to the pointwise layer
        2. The output size as specified in the configuration

        The input size includes:
        - Output from previous temporal stream
        - Output from previous pointwise layer
        - Original features and indicators
        - Static features

        Parameters:
            layer_idx: Index of the layer (0-based)

        Returns:
            Tuple of (input_size, output_size) for the pointwise transformation
        """
        # Calculate size of previous temporal output
        prev_temp_size = 0 if layer_idx == 0 else \
            (self.config.features + self._get_cumulative_features(layer_idx - 1)) * \
            self.config.temp_kernels[layer_idx - 1]

        # Calculate total input size including all features
        input_size = prev_temp_size + \
                     (self.config.point_sizes[layer_idx - 1] if layer_idx > 0 else 0) + \
                     2 * self.config.features + 1 + \
                     self.config.no_flat_features

        return input_size, self.config.point_sizes[layer_idx]

    def _get_final_input_size(self) -> int:
        """
        Calculate input size for final linear layer.

        This combines:
        1. The output from the last TPC layer
        2. The static features

        Returns:
            Size of the input to the final linear layer
        """
        return ((self.config.features + self._get_cumulative_features(self.config.num_layers)) *
                (1 + self.config.temp_kernels[-1]) +
                self.config.no_flat_features)

    def forward(self, x: Tensor, flat: Tensor, time_before_pred: int = 0) -> Tensor:
        """
        Forward pass of the TPC model.

        Parameters:
            x: Input tensor of shape [batch_size, seq_len, 2 * features + 1].
            flat: Static features tensor of shape [batch_size, no_flat_features].
            time_before_pred: Number of timesteps to exclude from the prediction.

        Returns:
            Output tensor of shape [batch_size, seq_len - time_before_pred, last_linear_size].
        """
        batch_size, seq_len, in_features = x.shape
        assert in_features == 2 * self.config.features + 1, f"Expected {2 * self.config.features + 1} input features, got {in_features}"

        # 1. Handle optional slicing for sequences with time_before_pred.
        effective_seq_len = seq_len - time_before_pred
        if time_before_pred > 0:
            # Remove earlier time points to enforce minimum observation window.
            x = x[:, time_before_pred:, :]
            seq_len = effective_seq_len

        # 2. Split input into temporal features (x_features) and indicators (x_indicators).
        x_features, x_indicators = torch.split(
            x[:, :, 1:],  # Exclude time column.
            self.config.features,
            dim=2,
        )  # Shape: [batch_size, seq_len, features].

        # 3. Static Features Preparation:
        # Flatten the static flat features repeated for all timesteps.
        # Shape: [(batch_size * seq_len), no_flat_features].
        x_flat = flat.repeat_interleave(seq_len, dim=0)
        # Reshape original input (used in pointwise layers).
        # Shape: [(batch_size * seq_len), 2F + 1].
        x_orig = x.reshape(batch_size * seq_len, -1)

        # 4. Initialize layer-wise states and skip connections:
        # Shape: [batch_size, 2 * features, seq_len].
        current_x = torch.stack([x_features, x_indicators], dim=2).reshape(
            batch_size, 2 * self.config.features, seq_len
        )
        # Pointwise skip connection starts from the features: [batch_size, features, seq_len].
        point_skip = x_features.permute(0, 2, 1)
        temp_output = None  # TempOutput from temporal stream (layer start).
        point_output = None  # PointOutput from pointwise stream (layer start).

        # 5. Process Through TPC Layers Sequentially:
        for i, layer in enumerate(self.layers):
            # TemporalPointwiseLayer Processes:
            # Inputs:
            #   - current_x: Temporal + indicator features with [B, 2F, T] shape at first layer.
            #   - point_skip: Updated pointwise skip [B, F, T].
            #   - prev_temp: Cumulative temp output from previous layers.
            #   - prev_point: Pointwise output cumulative.
            # Returns:
            # Updated versions of temp_output (temporal), point_output (pointwise),
            # current_x (next layer's input), and point_skip.
            temp_output, point_output, current_x, point_skip = layer(
                x=current_x,  # Current temporal input [B, (F + Zt) * (1 + Y), T].
                prev_temp=temp_output,  # Temporal output from prior layer.
                prev_point=point_output,  # Pointwise output from prior layer.
                point_skip=point_skip,  # Feature skip, updated each layer.
                batch_size=batch_size,
                seq_len=seq_len,
                padding=self._paddings[i],  # Pre-computed paddings for convs.
                temp_kernels=layer.temp_conv.out_channels // layer.temp_conv.in_channels,  # Temporal kernels.
                x_flat=x_flat,  # Static features repeated for all timesteps.
                x_orig=x_orig,  # Original input flattened.
            )

            # Comments on Progressing Dimensions:
            # - temp_output: Shape [(B * T), (F + Zt) * temp_kernels].
            # - point_output: Shape [(B * T), point_size[i]].
            # - current_x: Shape [B, ((F + Zt) * (1 + temp_kernels)), seq_len].
            # - point_skip: Updated features with skip connections [B, F + Zt, T].

        # 6. Combine Features for Final Prediction:
        # Combine static features (x_flat) and last-layer temporal stream (current_x).
        # current_x reshaped to [(B * (seq_len - time_before_pred)), all_features].
        combined_features = torch.cat(
            [
                x_flat,  # Shape: [(B * seq_len), no_flat_features].
                current_x.permute(0, 2, 1).reshape(batch_size * seq_len, -1),  # Flatten sequence.
            ],
            dim=1,
        )

        # Final Fully Connected Layer:
        # Produces output per timestep for each sample.
        return self.final_point(combined_features).reshape(
            batch_size, seq_len - time_before_pred, -1
        )

    @jit.script
    def _get_padding(conv_layer: nn.Conv1d) -> Tuple[int, int]:
        """
        Calculate padding size for a convolution layer.

        This calculates the padding needed to maintain the temporal dimension.
        The padding is applied only to the left side of the input.

        Parameters:
            conv_layer: Convolution layer to calculate padding for

        Returns:
            Tuple of (left_padding, right_padding) where right_padding is always 0
        """
        total_padding = (conv_layer.kernel_size[0] - 1) * conv_layer.dilation[0]
        return total_padding, 0