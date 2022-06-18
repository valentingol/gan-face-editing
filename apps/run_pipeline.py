"""Run translation and postprocessing pipeline."""

import os

from pipeline.depth_segmentation import depth_estimation_mix
from pipeline.domain_mixup import domain_mix
from pipeline.segmentation import segmentation_mix
from pipeline.translate import apply_translations
from pipeline.utils.global_config import GlobalConfig
from pipeline.utils.pipeline_paths import get_pipeline_paths


def check_environment():
    """Check if the environ is correctly set (forcing native GPU)."""
    assert 'FORCE_NATIVE' in os.environ \
        and os.environ['FORCE_NATIVE'] == '1', (
            'Please run the command: "FORCE_NATIVE=1 python3 apps/'
            'run_pipeline.py" instead!'
        )


def run(config):
    """Run the translation and postprocessing pipeline."""
    pipeline_paths = get_pipeline_paths(config)

    data_dir = pipeline_paths['data_dir']

    print('\nApplying translations in latent space...')
    apply_translations(**pipeline_paths['apply_translations'])

    if 'domain_mix' in pipeline_paths:
        print('\nApplying domain mixup...')
        domain_mix(data_dir=data_dir, **pipeline_paths['domain_mix'])

    if 'segmentation_mix' in pipeline_paths:
        print('\nApplying segmentation mixup...')
        segmentation_mix(data_dir=data_dir,
                         **pipeline_paths['segmentation_mix'])

    if 'depth_estimation_mix' in pipeline_paths:
        print('\nApplying depth estimation mixup...')
        depth_estimation_mix(data_dir=data_dir,
                             **pipeline_paths['depth_estimation_mix'])

    print('\nDone!')


if __name__ == '__main__':
    config = GlobalConfig.build_from_argv(fallback='configs/exp/base.yaml')
    check_environment()
    run(config)
