import os

if __name__ == '__main__':
    n_iter = 1000
    config = 'anycost-ffhq-config-f-flexible'
    output_dir = './data/anycost_flex'
    fails = []
    for img_name in os.listdir('data/input_images'):
        img_path = os.path.join('data/input_images', img_name)
        if img_name.endswith('.jpg'):
            try:
                os.system(f'FORCE_NATIVE=1 python tools/from_anycost_repo/'
                          f'project.py --config {config} '
                          f'--encoder --n_iter={n_iter} --enc_reg_weight=0.0 '
                          f'--output_dir={output_dir} {img_path}')
            except:
                fails.append(img_name)
    if fails != []:
        print('WARNING: failed to project: ', fails)
    else:
        print('SUCCESS: all images projected')
