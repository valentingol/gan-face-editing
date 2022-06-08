import os

from pipeline.translate import apply_translations
from pipeline.domain_mixup import domain_mix
from pipeline.segmentation import segmentation_mix
from pipeline.depth_segmentation import depth_estimation_mix

def get_config_pipeline():
    config_pipeline = {
        'data_dir': 'data/face_challenge',
        'apply_translations': {
            'proj_dir': 'projection/run1',
            'output_path': 'res/run1/images_post_translation',
            'config': 'anycost-ffhq-config-f',
            },
        'domain_mix': {
            'input_path': 'res/run1/images_post_translation',
            'output_path': 'res/run1/images_post_domain_mixup',
            'domains_dist_path': 'postprocess/domain_mixup/distances',
            'domains_img_path': 'postprocess/domain_mixup/domains'
            },
        'segmentation_mix': {
            'input_path': 'res/run1/images_post_domain_mixup',
            'output_path': 'res/run1/images_post_segmentation',
            'model_path': 'postprocess/segmentation/model/79999_iter.pth',
            },
        'depth_estimation_mix': {
            'input_path': 'res/run1/images_post_segmentation',
            'output_path': 'res/run1/images_post_depth_segmentation',
            'model_path': ('postprocess/depth_segmentation/model/'
                  'dpt_large-midas-2f21e586.pt'),
            },
        }
    return config_pipeline


if __name__ == '__main__':
    # Check if the environ is correctly set (forcing native GPU)
    assert 'FORCE_NATIVE' in os.environ and os.environ['FORCE_NATIVE'] == '1', (
        'Please run the command: "FORCE_NATIVE=1 python3 apps/'
        'run_pipeline.py" instead!'
        )

    config_pipeline = get_config_pipeline()
    data_dir = config_pipeline['data_dir']

    print('\nApplying translations in latent space...')
    apply_translations(**config_pipeline['apply_translations'])

    print('\nApplying domain mixup...')
    domain_mix(data_dir=data_dir, **config_pipeline['domain_mix'])

    print('\nApplying segmentation mixup...')
    segmentation_mix(data_dir=data_dir, **config_pipeline['segmentation_mix'])

    print('\nApplying depth estimation mixup...')
    depth_estimation_mix(data_dir=data_dir,
                         **config_pipeline['depth_estimation_mix'])

    print('Done!')
