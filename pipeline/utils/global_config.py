"""Global configuration file for the project."""

from rr.ml.config import Configuration


class GlobalConfig(Configuration):
    """Global configuration file for the project."""

    @staticmethod
    def get_default_config_path():
        """Get the default configuration path."""
        return 'configs/default/main.yaml'

    def parameters_pre_processing(self):
        """Pre-processing parameters."""
        return {'*_config_path': self.register_as_additional_config_file,
                'save_path': self.register_as_experiment_path}
