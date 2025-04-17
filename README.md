# Healthcare Prediction System
This repository contains a deep learning system for healthcare outcome prediction tasks using temporal data. The system leverages the Temporal Pointwise Convolution (TPC) architecture to predict various healthcare outcomes including length of stay (LOS), mortality, and next patient destination. We are applying this to the data in EPIC (the electronic health record system used in Cambridge University Hospitals NHS Trust).
## Project Structure
``` 
.
├── data/                   # Data directory (not included in repository)
│   ├── train.csv           # Training dataset
│   ├── val.csv             # Validation dataset
│   └── test.csv            # Test dataset
├── experiments/            # Experiment configuration
│   ├── base_config.yaml    # Base configuration file
│   └── results/            # Experiment results (generated during training)
├── src/                    # Source code
│   ├── data/               # Data loading and processing
│   │   └── dataset.py      # HealthcareDataModule implementation
│   ├── models/             # Model implementations
│   │   ├── lightning_module.py  # PyTorch Lightning module
│   │   ├── metrics.py      # Metric calculation utilities
│   │   └── tpc.py          # Temporal Pointwise Convolution implementation
│   ├── utils/              # Utility functions
│   │   └── hyperopt_utils.py # Hyperparameter optimization utilities
│   └── train.py            # Main training script
└── requirements.txt        # Project dependencies
```
## Setup
### Prerequisites
- Python 3.8+
- virtualenv

### Installation
1. Clone the repository:
``` bash
   git clone https://github.com/EmmaRocheteau/Cambridge_TPC.git
   cd Cambridge_TPC
```
1. Create and activate a virtual environment:
``` bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```
1. Install the required packages:
``` bash
   pip install -r requirements.txt
```
## Data Preparation
Place your dataset files in the directory with the following structure: `data/`
- : Training dataset `train.csv`
- : Validation dataset `val.csv`
- : Test dataset `test.csv`

_Note: Detailed data structure information will be provided once the final dataset is received._
## Running Experiments
The project uses PyTorch Lightning for training and experiment tracking. Experiments are configured through YAML files in the directory. `experiments/`
### Basic Training
To run a basic training experiment:
``` bash
python src/train.py --config experiments/base_config.yaml --experiment-name "my_experiment"
```
This will create a timestamped experiment directory under with the following structure: `experiments/results/`
``` 
experiments/results/YYYY-MM-DD_HHMMSS_my_experiment/
├── checkpoints/         # Model checkpoints
├── tensorboard/         # TensorBoard logs
└── config.yaml          # Copy of the experiment configuration
```
### Advanced Options
The training script supports several command-line arguments:
- : Path to the configuration file (default: ) `--config``experiments/base_config.yaml`
- : Name for the experiment `--experiment-name`
- : Enable hyperparameter optimization `--hyperopt`
- : Path to a checkpoint to resume training from `--checkpoint`
- : Override hardware accelerator (cpu, gpu, auto) `--accelerator`
- : Override number of devices to use `--devices`
- : Override distributed training strategy `--strategy`
- : Override training precision (16 or 32) `--precision`

Example with hardware configuration:
``` bash
python src/train.py --config experiments/base_config.yaml --experiment-name "gpu_training" --accelerator gpu --devices 2 --precision 16
```
## Hyperparameter Tuning
To run hyperparameter optimization:
``` bash
python src/train.py --config experiments/base_config.yaml --experiment-name "hyperopt_experiment" --hyperopt
```
This will use Ray Tune to search for optimal hyperparameters including:
- Number of model layers
- Dropout rates
- Learning rate
- Batch size
- Hidden dimensions
- Temporal kernel sizes

The optimization process will:
1. Run multiple trials with different hyperparameter configurations
2. Automatically terminate poorly performing trials using the ASHA scheduler
3. Generate a comprehensive analysis of parameter importance
4. Save the best configuration for future use

### Viewing Hyperparameter Optimization Results
During and after hyperparameter optimization, you can view the results:
1. Access the Ray Tune dashboard at [http://127.0.0.1:8265](http://127.0.0.1:8265)
2. Check the saved analysis files in `experiments/results/<timestamp>_hyperopt_experiment/hyperopt_results/`:
    - : Best hyperparameter configuration `best_config.yaml`
    - : Analysis of parameter importance `parameter_importance.csv`
    - : Correlations between parameters `parameter_correlations.csv`
    - : Data from all trials `all_trials.csv`
    - Interactive HTML visualizations of parameter importance and learning curves

## Experiment Tracking
### TensorBoard
You can monitor training progress using TensorBoard:
``` bash
tensorboard --logdir experiments/results/
```
Navigate to `http://localhost:6006` in your web browser to view:
- Training/validation losses
- Metrics for each task (AUROC, MSLE, accuracy, etc.)
- Learning rate changes
- Model architecture

### Finding the Best Experiment
To find the best experiment:
1. Launch TensorBoard pointing to the results directory:
``` bash
   tensorboard --logdir experiments/results/
```
1. In the TensorBoard interface:
    - Use the "Scalars" tab to compare metrics across experiments
    - Look for experiments with the lowest validation loss () `val_total_loss`
    - For task-specific performance, check individual metrics like or `val_mortality_auroc``val_los_prediction_msle`

2. The best model checkpoint for each experiment is saved in its `checkpoints/` directory with the filename pattern `best-model.ckpt`

## Configuration
The system is highly configurable through YAML configuration files. Key configuration sections include:
### Data Configuration
``` yaml
data:
  data_dir: "data/"
  train_file: "train.csv"
  val_file: "val.csv"
  test_file: "test.csv"
```
### Model Configuration
``` yaml
model:
  type: "TPC"  # Model type
  features: 64  # Number of features
  no_flat_features: 10  # Number of static features
  num_layers: 3
  dropout: 0.1
  temp_dropout_rate: 0.1
  # Additional model parameters...
```
### Task Configuration
``` yaml
tasks:
  los_prediction:
    type: "regression"
    loss_weight: 1.0
    metrics: ["msle", "mse", "mae", "mape", "r2"]
    regression_bins: [1, 2, 3, 4, 5, 6, 7, 8, 14]

  mortality:
    type: "binary"
    loss_weight: 1.0
    metrics: ["auroc", "auprc", "accuracy", "balanced_accuracy"]
    
  # Additional tasks...
```
### Training Configuration
``` yaml
training:
  seed: 42
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping:
    monitor: "val_total_loss"
    patience: 10
    mode: "min"
  # Additional training parameters...
```
### Hardware Configuration
The system supports various hardware configurations:
``` yaml
hardware:
  accelerator: "auto"  # Options: "cpu", "gpu", "auto"
  devices: "auto"      # Number of devices or "auto"
  strategy: "auto"     # Options: "ddp", "dp", null, "auto"
  precision: 32        # Options: 16, 32
```
These settings can be overridden via command-line arguments when running an experiment.
## Metrics
The system supports a comprehensive set of metrics for different types of prediction tasks:
### Regression Metrics (LOS Prediction)
- MSLE (Mean Squared Logarithmic Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Binned Accuracy (for discretized LOS ranges)

### Binary Classification Metrics (Mortality)
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Accuracy
- Balanced Accuracy
- F1 Score
- Precision
- Recall
- Specificity

### Multiclass Classification Metrics (Next Destination)
- Accuracy
- Balanced Accuracy
- Macro F1 Score
- Macro Precision
- Macro Recall
