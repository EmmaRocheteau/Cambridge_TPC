import argparse
from src.config.config import get_config
from src.data.dataset import load_data
from src.models.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = get_config(args.config)
    data = load_data(config.data_path)
    train_model(data, config)

if __name__ == "__main__":
    main()