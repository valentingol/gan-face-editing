import os
import sys

from pipeline.utils.global_config import GlobalConfig


def run(config):
    # Get the configs
    data_dir = config.data_dir
    output_dir = config.projection_dir
    n_iter = config.projection.n_iter
    enc_reg_weight = config.projection.enc_reg_weight
    mse_weight = config.projection.mse_weight
    optimize_sub_g = config.projection.optimize_sub_g

    flexible_config = config.anycost_config_flexible
    anycost_config = ('anycost-ffhq-config-f-flexible'
                      if flexible_config else 'anycost-ffhq-config-f')

    fails = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            try:
                command_line = ('FORCE_NATIVE=1 python anycostgan/tools/'
                                f'project.py --config {anycost_config} '
                                f'--encoder --n_iter={n_iter} '
                                f'--enc_reg_weight={enc_reg_weight} '
                                f'--mse_weight={mse_weight} '
                                f'--output_dir={output_dir} ')
                if optimize_sub_g:
                    command_line += '--optimize_sub_g '
                command_line += img_path

                # Project image
                os.system(command_line)

            except KeyboardInterrupt as e:
                print(e)
                sys.exit(1)
            except Exception as e:
                print(f'Error occurs while project image "{img_name}"')
                print(e)
                fails.append(img_name)

    if fails != []:
        print('WARNING: failed to project: ', fails)
    else:
        print('SUCCESS: all images projected')


if __name__ == '__main__':
    config = GlobalConfig.build_from_argv(fallback='configs/exp/base.yaml')
    run(config)
