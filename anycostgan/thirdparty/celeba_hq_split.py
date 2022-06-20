"""Celeba HQ dataset split utility."""


def get_celeba_hq_split():
    """Get the split of the Celeba HQ dataset."""
    train_idx = []
    test_idx = []
    val_idx = []

    with open(
            'thirdparty/CelebA-HQ-to-CelebA-mapping.txt', encoding='utf-8'
            ) as f:
        lines = f.readlines()
    celeba_ids = [int(line.strip().split()[1]) for line in lines]

    for idx, x in enumerate(celeba_ids):  # celeba-hq idx, celeba idx
        if 162771 <= x < 182638:
            val_idx.append(idx)
        elif x >= 182638:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    # NOTICE: the range of the index is 0-29999
    # but the range of the celebahq images fname is 1-30000
    return train_idx, val_idx, test_idx
