# Code from https://github.com/isl-org/DPT

""" Base model. """

import torch


class BaseModel(torch.nn.Module):
    """ Base model class. """
    def load(self, path):
        """Load model from file.

        Parameters
        ----------
        path : str
            File path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
