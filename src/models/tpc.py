from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass


@dataclass
class TPCConfig:
    """
    Configuration class for TPC model parameters.

    The TPC model processes temporal data with both time-varying and static features.
    This configuration holds all hyperparameters needed for the model architecture.

    Attributes:
        num_layers: Number of TPC layers in the model
        dropout: Dropout rate for pointwise (linear) layers
        temp_dropout_rate: Dropout rate for temporal convolution layers
        kernel_size: Size of the temporal convolution kernel
        temp_kernels: List of temporal kernel counts for each layer
        point_sizes: List of output sizes for pointwise layers in each layer
        momentum: Momentum parameter for batch normalization
        last_linear_size: Output size of the final linear layer
        features: Number of features (F) for temporal convolution
        no_flat_features: Number of static (flat) features
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

    These streams are then combined using skip connections to preserve both
    temporal and feature information.

    Key concepts:
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
        super().__init__()

        # Temporal convolution stream
        self.temp_conv = nn.Conv1d(
            in_channels=in_channels,  # (F + Zt) * (Y + 1)
            out_channels=in_channels * temp_kernels,  # (F + Zt) * Y
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels  # Each channel is convolved separately
        )

        # Now passing momentum parameter to BatchNorm layers
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
        1. Apply temporal convolution to capture temporal patterns
        2. Transform features using pointwise linear layer
        3. Combine results using skip connections

        Args:
            x: Input tensor containing temporal features
            prev_temp: Output from previous layer's temporal stream
            prev_point: Output from previous layer's pointwise stream
            point_skip: Skip connections for pointwise features
            batch_size: Number of samples in batch
            seq_len: Length of input sequences
            padding: Padding sizes for temporal convolution
            temp_kernels: Number of temporal kernels
            x_flat: Static features (if any)
            x_orig: Original input features

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
        concat_inputs = [t for t in [prev_temp, prev_point, x_orig, x_flat] if t is not None]
        x_concat = torch.cat(concat_inputs, dim=1)  # [(B * T), sum(feature_dims)]

        # Apply pointwise transformation
        point_output = self.dropout(
            self.bn_point(
                self.point_linear(x_concat)
            )
        )  # [(B * T), point_size[1]]

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
        x_combined = self.relu(
            torch.cat([temp_skip, x_point_rep], dim=1)  # [B, F + Zt + point_size[1], Y + 1, T]
        )

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

    This model processes temporal sequences by combining:
    1. Temporal convolutions to capture patterns across time
    2. Pointwise transformations to capture feature interactions
    3. Skip connections to maintain information flow
    """

    def __init__(self, config: TPCConfig):
        """
        Initialize the TPC model.

        Args:
            config: Configuration object containing model hyperparameters
        """
        super().__init__()
        self.config = config

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

        # Final linear layer for predictions
        self.final_point = nn.Linear(
            in_features=self._get_final_input_size(),
            out_features=config.last_linear_size
        )

    def _get_layer_channels(self, layer_idx: int) -> int:
        """
        Calculate input channels for a given layer.

        The number of channels increases with depth due to skip connections
        and additional features from pointwise layers.
        """
        if layer_idx == 0:
            return 2 * self.config.features
        return (self.config.features + self._get_cumulative_features(layer_idx)) * \
            (1 + self.config.temp_kernels[layer_idx - 1])

    def _get_dilation(self, layer_idx: int) -> int:
        """
        Calculate dilation for a given layer.

        Dilation increases with depth to capture longer-range temporal dependencies.
        """
        return layer_idx * (self.config.kernel_size - 1) if layer_idx > 0 else 1

    def _get_cumulative_features(self, layer_idx: int) -> int:
        """Calculate cumulative features added by pointwise layers up to a given layer"""
        return sum(self.config.point_sizes[:layer_idx])

    def _get_point_size(self, layer_idx: int) -> Tuple[int, int]:
        """
        Calculate input and output sizes for pointwise layer.

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
        """Calculate input size for final linear layer"""
        return ((self.config.features + self._get_cumulative_features(self.config.num_layers)) *
                (1 + self.config.temp_kernels[-1]) +
                self.config.no_flat_features)

    def forward(self, x: Tensor, flat: Tensor, time_before_pred: int = 0) -> Tensor:
        """
        Forward pass of the TPC model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, 2 * features + 1]
               Contains temporal features and their indicators
            flat: Static features tensor of shape [batch_size, no_flat_features]
            time_before_pred: Number of initial timesteps to exclude from prediction

        Returns:
            Output tensor of shape [batch_size, seq_len - time_before_pred, last_linear_size]
        """
        batch_size, seq_len, _ = x.shape

        # Split features and prepare initial tensors
        x_features, x_indicators = torch.split(
            x[:, :, 1:],  # Remove time feature
            self.config.features,
            dim=2
        )

        # Prepare repeated flat features and original input
        x_flat = flat.repeat_interleave(seq_len, dim=0)  # [(B * T), no_flat_features]
        x_orig = x.reshape(batch_size * seq_len, -1)  # [(B * T), 2F + 1]

        # Initialize layer states
        current_x = torch.stack([x_features, x_indicators], dim=2).reshape(
            batch_size,
            2 * self.config.features,
            seq_len
        )  # [B, 2F, T]
        point_skip = x_features.permute(0, 2, 1)  # [B, F, T]
        temp_output = None
        point_output = None

        # Process through TPC layers
        for layer in self.layers:
            temp_output, point_output, current_x, point_skip = layer(
                x=current_x,
                prev_temp=temp_output,
                prev_point=point_output,
                point_skip=point_skip,
                batch_size=batch_size,
                seq_len=seq_len,
                padding=self._get_padding(layer.temp_conv),
                temp_kernels=layer.temp_conv.out_channels // layer.temp_conv.in_channels,
                x_flat=x_flat,
                x_orig=x_orig
            )

        # Final prediction (excluding initial timesteps if specified)
        if time_before_pred > 0:
            current_x = current_x[:, :, time_before_pred:]
            x_flat = x_flat[time_before_pred:]

        # Combine all features for final prediction
        combined_features = torch.cat([
            x_flat,
            current_x.permute(0, 2, 1).reshape(batch_size * (seq_len - time_before_pred), -1)
        ], dim=1)

        return self.final_point(combined_features).reshape(
            batch_size,
            seq_len - time_before_pred,
            -1
        )

    @staticmethod
    def _get_padding(conv_layer: nn.Conv1d) -> Tuple[int, int]:
        """Calculate padding size for a convolution layer"""
        total_padding = (conv_layer.kernel_size[0] - 1) * conv_layer.dilation[0]
        return total_padding, 0