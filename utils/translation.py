import numpy as np
import torch

def get_translations(carac_list, flexible_config):
    """ Get translations for a given list of caracteristics.

    Parameters
    ----------
    carac_list : list[int]
        List of caracteristics (9 int elements).
    flexible_config : bool
        If True, configuration considered is AnyCost-flexible
        else it is vanilla AnyCost.

    Returns
    -------
    translations : dict[str, torch.Tensor (device cpu)]
        Translations for all the changes that are not in the original image
        (key: 'identifier', value: translation).
    """
    default_translations = default_translations(flexible_config)
    # TODO
    return


def default_translations(flexible_config):
    """ Get default translations.

    Parameters
    ----------
    flexible_config : bool
        If True, configuration considered is AnyCost-flexible
        else it is vanilla AnyCost.

    Returns
    -------
    translations : dict[str, torch.Tensor (device cpu)]
        Default translations for all the changes. (key: 'identifier',
        value: translation).
    """
    # TODO
    return
