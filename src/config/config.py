import yaml

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def get_config(config_path):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return Config(cfg)