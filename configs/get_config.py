from configs import base_config

def file_from_dataset(dataset_name):
    if dataset_name == "basedata":
        return base_config.get_default_configs()
    else:
        raise Exception("Dataset not defined.")