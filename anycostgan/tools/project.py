# Code from https://github.com/mit-han-lab/anycost-gan

""" Project image in the latent space. """

import argparse
import json
import os
import random

import lpips
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from anycostgan import models
from anycostgan.models.dynamic_channel import (CHANNEL_CONFIGS,
                                               set_uniform_channel_ratio,
                                               reset_generator,
                                               set_sub_channel_config)
from anycostgan.thirdparty import LBFGS
from anycostgan.utils.torch_utils import adaptive_resize


torch.backends.cudnn.benchmark = False


def make_image(tensor):
    """ Convert a tensor to a numpy uint8 image. """
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
        )


def extract_left_eye(img):
    """ Extract the left eye from an image. """
    if img.shape[-1] != 1024:
        img = F.interpolate(img, size=1024, mode='bilinear',
                            align_corners=True)
    return img[:, :, 270:485, 425:545]


def extract_right_eye(img):
    """ Extract the right eye from an image. """
    if img.shape[-1] != 1024:
        img = F.interpolate(img, size=1024, mode='bilinear',
                            align_corners=True)
    return img[:, :, 539:754, 425:545]


def compute_loss_sum(x, y, w):
    """ Compute the sum of all losses. """
    # WARNING: here we return the sum of losses for each sample,
    # so that the lr is not related to batch size

    # 1. compute the perceptual loss using 256 (or lower) resolution
    percep_loss = percept(adaptive_resize(x, resize),
                          adaptive_resize(y, resize)).sum()

    # 2. compute the mse loss on fully resolution
    # (empirically it is sharper)
    mse_loss = nn.MSELoss()(x, adaptive_resize(
        y, x.shape[-1]
        )) * x.shape[0] * args.mse_weight

    # 3. compute the encoder regularization loss
    if args.enc_reg_weight > 0.:
        # https://arxiv.org/pdf/2004.00049.pdf,
        # though the encoder is not trained with D
        assert encoder is not None
        pred_w = encoder(adaptive_resize(imgs, 256))
        enc_loss = nn.MSELoss()(w, pred_w) * x.shape[0] * args.enc_reg_weight
    else:
        enc_loss = torch.tensor(0.).to(device)

    loss = mse_loss + percep_loss + enc_loss
    # average loss per sample for display
    loss_dict = {'mse': mse_loss.item() / x.shape[0],
                 'lpips': percep_loss.item() / x.shape[0],
                 'encoder': enc_loss.item() / x.shape[0]}
    return loss, loss_dict


def process_generator():
    """ Process the generator. """
    if args.optimize_sub_g:
        if evolve_cfgs is not None:
            # The generator is trained with elastic channels and
            # evolved

            # Randomly pick an evolution config
            if random.random() < 0.5:
                rand_cfg = random.sample(list(evolve_cfgs.keys()), 1)
                set_sub_channel_config(generator, rand_cfg['channels'])
                generator.target_res = rand_cfg['res']
            else:
                reset_generator(generator)  # full G
        else:
            set_uniform_channel_ratio(generator,
                                      random.choice(CHANNEL_CONFIGS))
            generator.target_res = random.choice([256, 512, 1024])
    else:
        pass


def project_images(images, save_intermediate=False):
    """ Project images in the latent space. """
    with torch.no_grad():
        if encoder is not None:
            styles = encoder(adaptive_resize(images, 256))
        else:
            styles = generator.mean_style(10000).view(1, 1, -1).repeat(
                images.shape[0], generator.n_style, 1
                )

    input_kwargs = {'styles': styles, 'noise': None, 'randomize_noise': False,
                    'input_is_style': True}
    styles.requires_grad = True

    # we only optimize the styles but not noise; with noise
    # it is harder to manipulate
    if args.optimizer.lower() == "lbfgs":
        optimizer = LBFGS.FullBatchLBFGS([styles], lr=1)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam([styles], lr=0.001)
    else:
        raise NotImplementedError

    with torch.no_grad():
        init_image = generator(**input_kwargs)[0].clamp(-1, 1)
        loss, loss_dict = compute_loss_sum(init_image, images, styles)

    loss_list = []
    loss_list.append(loss_dict)
    pbar = tqdm(range(args.n_iter))
    for iteration in pbar:
        if save_intermediate and (iteration + 1) % 100 == 0:
            proj_styles = styles.detach()
            with torch.no_grad():
                rec_images = generator(proj_styles, randomize_noise=False,
                                       input_is_style=True)[0]
                img_ar = make_image(rec_images)
                for i, img in enumerate(img_ar):
                    pil_img = Image.fromarray(img)
                    pil_img.save(f'img_{i}_iter_{iteration + 1}.png')

        if isinstance(optimizer, LBFGS.FullBatchLBFGS):
            def closure():
                optimizer.zero_grad()
                process_generator()
                out = generator(**input_kwargs)[0].clamp(-1, 1)
                reset_generator(generator)
                loss, loss_dict = compute_loss_sum(out, images, styles)
                loss_list.append(loss_dict)
                return loss

            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, _ = optimizer.step(options=options)
        else:
            process_generator()
            out = generator(**input_kwargs)[0]
            reset_generator(generator)
            loss, loss_dict = compute_loss_sum(out, images, styles)
            loss.backward()
            optimizer.step()
            loss_list.append(loss_dict)
        pbar.set_postfix(loss_list[-1])
    return styles.detach()


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
        )
    # NOTE configs : 'anycost-ffhq-config-f',
    # 'anycost-ffhq-config-f-flexible', 'stylegan2-ffhq-config-f'
    parser.add_argument("--config", type=str, default='anycost-ffhq-config-f',
                        help="models config")
    parser.add_argument("--encoder", action="store_true",
                        help="use encoder prediction as init")
    # NOTE optimizer : 'lbfgs', 'adam'
    parser.add_argument("--optimizer", type=str, default='lbfgs',
                        help="optimizer used")
    parser.add_argument("--n_iter", type=int, default=100,
                        help="optimize iterations")
    parser.add_argument("--optimize_sub_g", action="store_true",
                        help="also optimize the sub-generators")
    # loss weight
    parser.add_argument("--mse_weight", type=float, default=1.0,
                        help="weight of MSE loss")
    parser.add_argument("--enc_reg_weight", type=float, default=0.0,
                        help="weight of encoder regularization loss")
    # save intermediate images (every 100 iterations)
    parser.add_argument("--save_intermediate", action='store_true',
                        help="save intermediate images every 100 iterations")
    # output directory to save images and latent inside
    parser.add_argument("--output_dir", type=str, default='./',
                        help="output directory to save images and latent "
                        "inside")
    # file list (sep with space)
    parser.add_argument("files", metavar="FILES", nargs="+",
                        help="path to image files to be projected")

    args = parser.parse_args()

    n_mean_latent = 10000

    # build generator to project
    generator = models.get_pretrained('generator', args.config).to(device)

    generator.eval()
    if args.encoder:
        encoder = models.get_pretrained('encoder', args.config).to(device)
        encoder.eval()
    else:
        encoder = None

    # if the generator is trained with elastic channels and evolution
    # search
    if 'flexible' in args.config:
        print(' * loading evolution configs...')
        with open(os.path.join(
                f'anycostgan/evolve_configs/{args.config}.json'
                )) as f:
            evolve_cfgs = json.load(f)
        # pick some reduction ratios; you can modify this to include
        # more or fewer reduction ratio: search MACs limit (the key in
        # evolve cfgs)
        cfg_map = {
            '2x': '73G',
            '4x': '36G',
            '6x': '24G',
            '8x': '18G',
            '10x': '15G',
        }
        evolve_cfgs = {k: evolve_cfgs[v] for k, v in cfg_map.items()}
    else:
        evolve_cfgs = None

    # load perceptual loss
    percept = lpips.LPIPS(net='vgg', verbose=False).to(device)

    # load images to project
    resize = min(generator.resolution, 256)
    transform = transforms.Compose([
        transforms.Resize(generator.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    imgs = [transform(Image.open(f).convert("RGB")) for f in args.files]
    imgs = torch.stack(imgs, 0).to(device)
    projected_styles = project_images(imgs,
                                      save_intermediate=args.save_intermediate)

    with torch.no_grad():
        rec_images = generator(projected_styles, randomize_noise=False,
                               input_is_style=True)[0]

    img_ar = make_image(rec_images)

    result_file = {}
    if not os.path.exists(os.path.join(args.output_dir, 'projected_images')):
        os.makedirs(os.path.join(args.output_dir, 'projected_images'))
    if not os.path.exists(os.path.join(args.output_dir, 'projected_latents')):
        os.makedirs(os.path.join(args.output_dir, 'projected_latents'))

    for i, input_name in enumerate(args.files):
        result_file[input_name] = {
            "img": rec_images[i],
            "latent": projected_styles[i],
        }
        basename = os.path.splitext(os.path.basename(input_name))[0]
        path_img = os.path.join(args.output_dir, 'projected_images', basename)
        img_name = path_img + "-project.png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

        path_latent = os.path.join(args.output_dir, 'projected_latents',
                                   basename)
        np.save(path_latent + '.npy', projected_styles[i].cpu().numpy())
