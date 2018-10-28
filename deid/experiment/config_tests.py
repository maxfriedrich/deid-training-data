from .config import Config


def example_config():
    return Config({'a': 0, 'b': 1, 'c': {'d': 2}})


def test_config_behaves_like_a_dict():
    config = example_config()
    assert config['a'] == 0
    assert config['b'] == 1
    assert config['c']['d'] == 2

    config['c']['d'] = 3
    assert config['c']['d'] == 3


def test_config_returns_none_for_missing_values():
    config = example_config()
    assert config['x'] is None
    assert config['c']['y'] is None
