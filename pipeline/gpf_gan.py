import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer


def apply_gfp_gan(input_path : str, output_path : str, model_path : str, new_config = {}):

    # ------------------------ update configs ------------------------
    configs = { 
        "version" : "1.3",
        "upscale" : 2,
        "bg_upsampler" : "realesrgan",
        "bg_tile" : 400, #0 during testing
        "suffix":  None,
        "only_center_face" : "store_true",
        "aligned" : "store_true", #action
        "ext" : "auto"
    }

    configs.update(new_config)
   

    # ------------------------ input & output ------------------------
    if input_path.endswith('/'):
        input_path = input_path[:-1]
    if os.path.isfile(input_path):
        img_list = [input_path]
    else:
        #raise NameError("The input path doesn't exist !")
        img_list = sorted(glob.glob(os.path.join(input_path, '*')))

    os.makedirs(output_path, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if configs["bg_upsampler"] == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=configs["bg_tile"],
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode #we do not go there if there is no GPU in all case
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    # WE DO NOT IMPLEMENT OTHER VERSIONS FOR NOW

    # if args.version == '1':
    #     arch = 'original'
    #     channel_multiplier = 1
    #     model_name = 'GFPGANv1'
    # elif args.version == '1.2':
    #     arch = 'clean'
    #     channel_multiplier = 2
    #     model_name = 'GFPGANCleanv1-NoCE-C2'
    
    if configs["version"] == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
    else:
        raise ValueError(f'Wrong model version {configs["version"]}.')

    # determine model paths : below is useful to handle several models but we only have one and precise it in argument
    #model_path = os.path.join('../models/gfp_gan', model_name + '.pth')
    # if not os.path.isfile(model_path):
    #     model_path = os.path.join('realesrgan/weights', model_name + '.pth')

    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    restorer = GFPGANer(
        model_path=model_path,
        upscale=configs["upscale"],
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)


    # ------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=configs['aligned'], only_center_face=configs["only_center_face"], paste_back=True)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(output_path, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if configs["suffix"] is not None:
                save_face_name = f'{basename}_{idx:02d}_{configs["suffix"]}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(output_path, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(output_path, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if configs['ext'] == 'auto':
                extension = ext[1:]
            else:
                extension = configs['ext']

            if configs["suffix"] is not None:
                save_restore_path = os.path.join(output_path, 'restored_imgs', f'{basename}_{configs["suffix"]}.{extension}')
            else:
                save_restore_path = os.path.join(output_path, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{output_path}] folder.')


if __name__ == '__main__':
    print("Applying GFP GAN ...")
    INPUT_PATH = "data/input_gfp_gan"
    OUTPUT_PATH = "data/output_gfp_gan"
    MODEL_PATH = "models/gfp_gan/GFPGANv1.3.pth"
    #Precise custom configs if you want to change the default values : empty if no modifications
    #if not precised, it is empty by default ...
    MODIF_CONFIG = {}
    
    apply_gfp_gan(input_path = INPUT_PATH, output_path = OUTPUT_PATH, model_path = MODEL_PATH, new_config = MODIF_CONFIG)


