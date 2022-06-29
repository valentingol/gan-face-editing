"""Utilities for pipeline paths."""

from os.path import join


def get_pipeline_paths(config):
    """Get the configs for the pipeline."""
    save_intermediate = config.pipeline.save_intermediate
    skip_gfp_gan = config.pipeline.skip_gfp_gan
    skip_domain_mixup = config.pipeline.skip_domain_mixup
    skip_segmentation = config.pipeline.skip_segmentation
    skip_depth_segmentation = config.pipeline.skip_depth_segmentation

    projection_dir = config.projection_dir
    result_dir = config.result_dir
    config_flexible = config.anycost_config_flexible
    anycost_config = ('anycost-ffhq-config-f-flexible'
                      if config_flexible else 'anycost-ffhq-config-f')

    pipeline_paths = {
        'data_dir': config.data_dir,
        'apply_translations': {
            'projection_dir': projection_dir,
            'output_path': join(result_dir, 'images_post_translation'),
            'anycost_config': anycost_config,
            'configs': config.translation,
        },
        'gfp_gan_mix': {
            'model_path': 'postprocess/gfp_gan/model/GFPGANv1.3.pth',
            'configs': config.gfp_gan,
        },
        'domain_mix': {
            'domains_dist_path': 'postprocess/domain_mixup/distances',
            'domains_img_path': 'postprocess/domain_mixup/domains',
            'configs': config.domain_mixup,
        },
        'segmentation_mix': {
            'model_path': ('postprocess/segmentation/model/79999_iter'
                           '.pth'),
            'configs': config.segmentation,
        },
        'depth_estimation_mix': {
            'model_path': ('postprocess/depth_segmentation/model/'
                           'dpt_large-midas-2f21e586.pt'),
            'configs':
            config.depth_segmentation,
        },
    }

    op_order = [
        'apply_translations', 'gfp_gan_mix', 'domain_mix', 'segmentation_mix',
        'depth_estimation_mix'
    ]

    if skip_gfp_gan:
        op_order.remove('gfp_gan_mix')
    if skip_domain_mixup:
        op_order.remove('domain_mix')
        del pipeline_paths['domain_mix']
    if skip_segmentation:
        op_order.remove('segmentation_mix')
        del pipeline_paths['segmentation_mix']
    if skip_depth_segmentation:
        op_order.remove('depth_estimation_mix')
        del pipeline_paths['depth_estimation_mix']

    pipeline_paths = set_paths(pipeline_paths, op_order, result_dir,
                               save_intermediate)

    return pipeline_paths


def set_paths(pipeline_paths, op_order, result_dir, save_intermediate):
    """Set input path and output path for each operation."""
    for op_i, op_name in enumerate(op_order):
        if op_i != 0:  # not apply_translations
            if save_intermediate:
                prev_op = op_order[op_i - 1]
                for op_n, key_path in zip([prev_op, op_name],
                                          ['input_path', 'output_path']):
                    if op_n == 'apply_translations':
                        pipeline_paths[op_name][key_path] = join(
                            result_dir, 'images_post_translation')
                    if op_n == 'gfp_gan_mix':
                        pipeline_paths[op_name][key_path] = join(
                            result_dir, 'images_post_gfp_gan')
                    if op_n == 'domain_mix':
                        pipeline_paths[op_name][key_path] = join(
                            result_dir, 'images_post_domain_mixup')
                    if op_n == 'segmentation_mix':
                        pipeline_paths[op_name][key_path] = join(
                            result_dir, 'images_post_segmentation')
                    if op_n == 'depth_estimation_mix':
                        pipeline_paths[op_name][key_path] = join(
                            result_dir, 'images_post_depth_segmentation')
            else:
                pipeline_paths[op_name]['input_path'] = join(
                    result_dir, 'output_images')
                pipeline_paths[op_name]['output_path'] = join(
                    result_dir, 'output_images')
        if op_i == 0 and not save_intermediate:
            pipeline_paths['apply_translations']['output_path'] = join(
                result_dir, 'output_images')
    return pipeline_paths
