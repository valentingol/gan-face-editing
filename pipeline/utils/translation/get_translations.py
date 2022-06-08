import os

import numpy as np
import torch


def browse_translation_dir(translation_dir, caracs):
    translations = {}
    translations_default = {}
    for fname in os.listdir(translation_dir):
        translation_path = os.path.join(translation_dir, fname)
        basename = fname.split('.')[0]
        split_name = basename.split('_')
        char = split_name[0]
        if len(split_name) == 2:
            if split_name[1] not in {'min', 'max'}:
                num = int(split_name[1])
                if caracs[char] != num:
                    transfo = np.load(translation_path)
                    translations_default[char + '_' + str(num)] = torch.tensor(
                        transfo, dtype=torch.float32
                        )
            else:
                degree = split_name[1]
                transfo = np.load(translation_path)
                translations_default[char + '_' + degree] = torch.tensor(
                    transfo, dtype=torch.float32
                    )
        else:
            if split_name[1] not in {'min', 'max'}:
                num, from_num = int(split_name[1]), int(split_name[3])
                if caracs[char] == from_num:
                    transfo = np.load(translation_path)
                    translations[char + '_' + str(num)] = torch.tensor(
                        transfo, dtype=torch.float32
                        )
            else:
                raise NotImplementedError('Specific translation of "cursor '
                                          'attribute" is not supported yet.')
    # Overwrite default translations with translations
    # from specific caracteristics
    translations = {**translations_default, **translations}
    return translations


def get_translations(carac_list, translation_dir, use_precomputed=True):
    """ Get translations for a given list of caracteristics.

    Parameters
    ----------
    carac_list : list[int]
        List of caracteristics (9 int elements).
    translation_dir: str
        Path to the directory containing the translations.

    Returns
    -------
    translations : dict[str, torch.Tensor (device cpu)]
        Translations for all the changes that are not in the original image
        (key: 'identifier', value: translation).
    """
    sk, a, se, _, _,  b, hc, d, hs = carac_list
    caracs = {'Sk': sk, 'A': a, 'Se': se, 'B': b, 'Hc': hc, 'D': d, 'Hs': hs}

    # Get pre-computed translations
    if use_precomputed:
        precomputed_translations = browse_translation_dir(
            'projection/default_translations', caracs
            )
    else:
        precomputed_translations = {}

    # Get custom translations
    translations = browse_translation_dir(translation_dir, caracs)
    translations = {**precomputed_translations, **translations}

    # Replace 'Hc_4' by 'bald':
    if hc != 4:
        translation_bald = translations.pop('Hc_4')
        translations['bald'] = translation_bald

    # Check if there is enough translations
    if hc != 4:  # not bald
        assert len(translations) == 24, (
            f'The number of translations is {len(translations)} while it '
            'should be 24.'
            )
    else:  # bald
        assert len(translations) == 25, (
            f'The number of translations is {len(translations)} while it '
            'should be 25 (for bald people).'
            )

    return translations
