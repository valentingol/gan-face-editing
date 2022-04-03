import os

if __name__ == '__main__':

    input_dir = 'submissions/submission4'
    sub_name = input_dir.split('/')[-1]
    out_dir = f'submissions/{sub_name}_per_carac'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n_img = len(os.listdir(input_dir))
    for i, dirname in enumerate(os.listdir(input_dir)):
        dirpath = os.path.join(input_dir, dirname)
        for fname in os.listdir(dirpath):
            carac = fname.split('.')[0]
            if not os.path.exists(os.path.join(out_dir, carac)):
                os.mkdir(os.path.join(out_dir, carac))
            src_path = os.path.join(dirpath, fname)
            target_path = os.path.join(out_dir, carac, dirname + '.png')
            os.system(f'cp {src_path} {target_path}')
        print(f'image {i+1}/{n_img} moved  ', end='\r')
    print()
