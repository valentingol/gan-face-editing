import os
from os.path import join

from pipeline.utils.global_config import GlobalConfig
from pipeline.translate import apply_translations
from pipeline.domain_mixup import domain_mix
from pipeline.segmentation import segmentation_mix
from pipeline.depth_segmentation import depth_estimation_mix


def get_config_pipeline(config):
    save_intermediate = config.pipeline.save_intermediate
    skip_domain_mixup = config.pipeline.skip_domain_mixup

    projection_dir = config.projection_dir
    result_dir = config.result_dir
    config_flexible = config.anycost_config_flexible
    anycost_config = ('anycost-ffhq-config-f-flexible' if config_flexible
                      else 'anycost-ffhq-config-f')

    config_pipeline = {
        'data_dir': config.data_dir,
        'apply_translations': {
            'projection_dir': projection_dir,
            'output_path': join(result_dir, 'images_post_translation'),
            'anycost_config': anycost_config,
            'configs': config.translation,
            },
        'domain_mix': {
            'input_path': join(result_dir, 'images_post_translation'),
            'output_path': join(result_dir, 'images_post_domain_mixup'),
            'domains_dist_path': 'postprocess/domain_mixup/distances',
            'domains_img_path': 'postprocess/domain_mixup/domains',
            'configs': config.domain_mixup,
            },
        'segmentation_mix': {
            'input_path': join(result_dir, 'images_post_domain_mixup'),
            'output_path': join(result_dir, 'images_post_segmentation'),
            'model_path': 'postprocess/segmentation/model/79999_iter.pth',
            'configs': config.segmentation,
            },
        'depth_estimation_mix': {
            'input_path': join(result_dir, 'images_post_segmentation'),
            'output_path': join(result_dir, 'images_post_depth_'
                                'segmentation'),
            'model_path': ('postprocess/depth_segmentation/model/'
                           'dpt_large-midas-2f21e586.pt'),
            'configs': config.depth_segmentation,
            },
        }

    if not save_intermediate:
        output_path = join(result_dir, 'output_images')
        config_pipeline['apply_translations']['output_path'] = output_path
        config_pipeline['domain_mix']['input_path'] = output_path
        config_pipeline['domain_mix']['output_path'] = output_path
        config_pipeline['segmentation_mix']['input_path'] = output_path
        config_pipeline['segmentation_mix']['output_path'] = output_path
        config_pipeline['depth_estimation_mix']['input_path'] = output_path
        config_pipeline['depth_estimation_mix']['output_path'] = output_path

    if skip_domain_mixup:
        del config_pipeline['domain_mix']
        config_pipeline['segmentation_mix']['input_path'] \
            = config_pipeline['apply_translations']['output_path']

    return config_pipeline


def check_environment():
    """Check if the environ is correctly set (forcing native GPU)."""
    assert 'FORCE_NATIVE' in os.environ \
        and os.environ['FORCE_NATIVE'] == '1', (
            'Please run the command: "FORCE_NATIVE=1 python3 apps/'
            'run_pipeline.py" instead!'
        )


def run(config):
    config_pipeline = get_config_pipeline(config)
    skip_domain_mixup = config.pipeline.skip_domain_mixup
    data_dir = config_pipeline['data_dir']

    print('\nApplying translations in latent space...')
    apply_translations(**config_pipeline['apply_translations'])

    if not skip_domain_mixup:
        print('\nApplying domain mixup...')
        domain_mix(data_dir=data_dir, **config_pipeline['domain_mix'])

    print('\nApplying segmentation mixup...')
    segmentation_mix(data_dir=data_dir, **config_pipeline['segmentation_mix'])

    print('\nApplying depth estimation mixup...')
    depth_estimation_mix(data_dir=data_dir,
                         **config_pipeline['depth_estimation_mix'])

    print('Done!')


if __name__ == '__main__':
    config = GlobalConfig.build_from_argv(fallback='configs/exp/base.yaml')
    check_environment()
    run(config)
