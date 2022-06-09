from rr.ml.config import Configuration


class GlobalConfig(Configuration):
    @staticmethod
    def get_default_config_path():
        return 'configs/default/main.yaml'

    def parameters_pre_processing(self):
        return {'*_config_path': self.register_as_additional_config_file,
                'save_path': self.register_as_experiment_path}
