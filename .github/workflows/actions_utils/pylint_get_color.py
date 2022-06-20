"""Manage badge color."""

import sys
from colorsys import hsv_to_rgb


def score_to_rgb_color(score, score_min, score_max):
    """Convert score to rgb color."""
    norm_score = max(0, (score-score_min) / (score_max-score_min))
    hsv = (1 / 3 * norm_score, 1, 1)
    rgb = hsv_to_rgb(*hsv)
    rgb = tuple(int(255 * value) for value in rgb)
    return f"rgb{rgb}"


if __name__ == '__main__':
    SCORE_MIN = 7.0
    SCORE_MAX = 10.0

    arg = sys.argv[1]
    score = float(arg.split('=')[1])

    if score < SCORE_MIN:
        raise ValueError(
                f'Pylint score {score} is lower than'
                f'minimum ({SCORE_MIN})'
                )

    print(score_to_rgb_color(score, SCORE_MIN, SCORE_MAX))
