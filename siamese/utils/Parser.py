import os
import json
from dotmap import DotMap


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = DotMap(config_dict)
    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join(
        "experiments", config.exp.project_name, "summary/")
    config.checkpoint_dir = os.path.join(
        "experiments", config.exp.project_name, "checkpoint/")
    return config
