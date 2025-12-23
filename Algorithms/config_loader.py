import yaml


def load_yaml(file_name):
    with open(f'./{file_name}') as file:
        config = yaml.safe_load(file)
    return config


def get_behavior_settings(config):
    behavior = next(iter(config['behaviors']))
    return config['behaviors'][behavior]


def get_trainer_type(config):
    behavior_settings = get_behavior_settings(config)
    return behavior_settings['trainer_type']


def get_hyperparameters(behave_settings):
    return behave_settings['hyperparameters']


def get_network_settings(behave_settings):
    return behave_settings['network_settings']


def get_reward_signals(behave_settings):
    return behave_settings['reward_signals']


def get_max_steps(behave_settings):
    return behave_settings['max_steps']


def get_environment_parameters(config):
    return config.get('environment_parameters',{})
