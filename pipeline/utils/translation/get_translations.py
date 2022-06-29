"""Get the translation to apply to the images."""

import os

import numpy as np
import torch


def browse_translation_dir(translation_dir, characs):
    """Browse the translation directory and get the translations."""
    translations = {}
    translations_default = {}

    # If no translations are provided, skip the browsing
    if not os.path.exists(translation_dir):
        return {}

    if characs is None:  # Apply all transformations
        for fname in os.listdir(translation_dir):
            translation_path = os.path.join(translation_dir, fname)
            transfo = np.load(translation_path)
            basename = fname.split('.')[0]
            translations[basename] = torch.tensor(transfo, dtype=torch.float32)
    else:  # Apply only the transformations that are not already present
        for fname in os.listdir(translation_dir):
            translation_path = os.path.join(translation_dir, fname)
            basename = fname.split('.')[0]
            split_name = basename.split('_')
            char = split_name[0]
            if len(split_name) == 2:
                if split_name[1] not in {'min', 'max'}:
                    num = int(split_name[1])
                    if characs[char] != num:
                        transfo = np.load(translation_path)
                        translations_default[char + '_' + str(num)] \
                            = torch.tensor(transfo, dtype=torch.float32)
                else:
                    degree = split_name[1]
                    transfo = np.load(translation_path)
                    translations_default[char + '_' + degree] = torch.tensor(
                        transfo, dtype=torch.float32)
            else:
                if split_name[1] not in {'min', 'max'}:
                    num, from_num = int(split_name[1]), int(split_name[3])
                    if characs[char] == from_num:
                        transfo = np.load(translation_path)
                        translations[char + '_' + str(num)] = torch.tensor(
                            transfo, dtype=torch.float32)
                else:
                    raise NotImplementedError(
                        'Specific translation (ie. using "fr_...") of '
                        '"min/max attribute" is not supported yet. '
                        f'{translation_path} should be removed.')

        # Overwrite default translations with translations
        # from specific characteristics
        translations = {**translations_default, **translations}
    return translations


def get_translations(translation_dir, charac_list, use_precomputed):
    """Get translations for a given list of characteristics.

    Parameters
    ----------
    translation_dir: str
        Path to the directory containing the translations.
    charac_list : list[int] or None
        If not None, charac_list should be list of characteristics with 9
        int elements following the convention presented here:
        https://transfer-learning.org/rules. Then, the translations
        will be based on the characteristics provided. If None,
        no characteristics are taken into account and the translations
        will be always the same (unconditional).
    use_precomputed : bool
        If True, use the precomputed translations in
        'projection/default_translations' and overwrite with
        translations in translation_dir. If False, compute only
        the translations in translation_dir.

    Returns
    -------
    translations : dict[str, torch.Tensor (device cpu)]
        Translations for all the changes that are not in the original image
        (key: 'identifier', value: translation).
    """
    if charac_list is not None:
        (skin, age, sex, _, _, bang, haircolor, doublechin,
         hairstyle) = charac_list
        characs = {
            'Sk': skin,
            'A': age,
            'Se': sex,
            'B': bang,
            'Hc': haircolor,
            'D': doublechin,
            'Hs': hairstyle
        }
    else:
        characs = None

    # Get pre-computed translations
    if use_precomputed:
        precomputed_translations = browse_translation_dir(
            'projection/default_translations/unconditional', characs)
        # Add conditional translations only when
        # characteristics are given
        if characs is not None:
            precomputed_translations_2 = browse_translation_dir(
                'projection/default_translations/conditional', characs)
            precomputed_translations = {
                **precomputed_translations,
                **precomputed_translations_2
            }
    else:
        precomputed_translations = {}

    # Get custom translations
    translations = browse_translation_dir(translation_dir, characs)
    translations = {**precomputed_translations, **translations}

    if characs is not None:
        # Replace 'Hc_4' by 'bald':
        if haircolor != 4:
            translation_bald = translations.pop('Hc_4')
            translations['bald'] = translation_bald

    return translations
