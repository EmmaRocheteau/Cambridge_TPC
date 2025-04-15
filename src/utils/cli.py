import yaml
from pathlib import Path

def list_experiments(log_dir="logs"):
    print(f"Experiments in {log_dir}:")
    for exp in Path(log_dir).glob("*/events.out.tfevents.*"):
        print(f"- {exp.parent.name}")

def load_config_for_experiment(log_dir, experiment_name):
    config_path = Path(log_dir) / experiment_name / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"No config.yaml found in {config_path}")

def rerun_with_config(config):
    from src.models.train import train_model
    from src.data.dataset import load_data
    from src.config.config import Config
    config_obj = Config(config)
    data = load_data(config_obj.data_path)
    train_model(data, config_obj)

if __name__ == "__main__":
    list_experiments()