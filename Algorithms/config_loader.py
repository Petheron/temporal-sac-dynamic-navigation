import yaml

class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.YAMLError(
                    f"Duplicate Key {key!r} at Line {key_node.start_mark.line + 1}. "
                    f"PyYAML Would Silently Discard the Earlier Block."
                )
            seen.add(key)
        return super().construct_mapping(node, deep)

def load_yaml(file_name):
    with open(f'./{file_name}') as file:
        config = yaml.load(file, Loader=_UniqueKeyLoader)
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
