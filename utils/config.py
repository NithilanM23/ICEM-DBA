import os
import yaml

DEFAULT_CONFIG = {
    'MODE': 1,
    'SEED': 10,
    'GPU': [0],
    'PATH': './checkpoints',
    'RESULTS': './results',
    'INPUT_SIZE': 256,
    'DIM': 48,
    'NUM_BLOCKS': [4, 6, 6, 8],
    'NUM_REFINEMENT_BLOCKS': 4,
    'D_STATE': 16,
    'INNER_RANK': 32,
    'D_KEY': 16,
    'FFN_EXPANSION_FACTOR': 2,
    'BATCH_SIZE': 4,
    'LR': 0.0001,
    'D2G_LR': 0.1,
    'BETA1': 0.9,
    'BETA2': 0.999,
    'LR_DECAY': 0.5,
    'MILESTONES': [20000, 40000, 60000, 80000, 120000],
    'MAX_ITERS': 4000000,
    'L1_LOSS_WEIGHT': 1.0,
    'STYLE_LOSS_WEIGHT': 250.0,
    'CONTENT_LOSS_WEIGHT': 0.1,
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,
    'TV_LOSS_WEIGHT': 0.0,
    'BSD_LOSS_WEIGHT': 1.0,
    'GAN_LOSS': 'lsgan',
    'GRAD_ACCUM_STEPS': 1,
    'USE_CHECKPOINT': False,
    'AMP': False,
    'PATCH_SIZE': 4,
    'SAVE_INTERVAL': 1000,
    'SAMPLE_INTERVAL': 500,
    'LOG_INTERVAL': 20,
    'VERBOSE': 1
}

class Config(dict):
    """Configuration dictionary with attribute-style access and YAML loading."""
    def __init__(self, config_path=None, **kwargs):
        super().__init__()
        self._dict = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self._dict.update(loaded)
            self._dict['CONFIG_PATH'] = config_path
            self._dict['PATH'] = os.path.dirname(config_path)

        for k, v in kwargs.items():
            if v is not None:
                self._dict[k] = v

    def __getattr__(self, name):
        if name in self._dict and self._dict[name] is not None:
            return self._dict[name]
        if name in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._dict[name] = value

    def __getitem__(self, item):
        return self.__getattr__(item)

    def print(self):
        print("=" * 50)
        print("ICEM-DBA Model Configuration:")
        print("=" * 50)
        for k, v in sorted(self._dict.items()):
            print(f"  {k}: {v}")
        print("=" * 50)
