import os

if __name__ == '__main__':
    data_dir = 'data/face_challenge'
    output_dir = 'projection/run1'
    n_iter = 1000
    enc_reg_weight = 0.0
    config = 'anycost-ffhq-config-f'

    fails = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            try:
                os.system(f'FORCE_NATIVE=1 python anycostgan/tools/'
                          f'project.py --config {config} '
                          f'--encoder --n_iter={n_iter} '
                          f'--enc_reg_weight={enc_reg_weight} '
                          f'--output_dir={output_dir} {img_path}')
            except:
                fails.append(img_name)
    if fails != []:
        print('WARNING: failed to project: ', fails)
    else:
        print('SUCCESS: all images projected')
