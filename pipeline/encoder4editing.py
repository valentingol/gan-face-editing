"""Process dataset."""

import os
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pipeline.utils.encoder4editing.models.psp import pSp
from pipeline.utils.segmentation import infer

# Load face parsing network
parsing_net = infer.load_model()

# Load e4e model
MODEL_PATH = "postprocess/encoder4editing/model/e4e_ffhq_encode.pt"
resize_dims = (256, 256)
# Setup required image transformations
img_transforms = transforms.Compose([
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

ckpt = torch.load(MODEL_PATH, map_location='cpu')
for u in list(ckpt.keys()):
    if "opts" not in u:
        ckpt.pop(u)
opts = ckpt['opts']
opts['checkpoint_path'] = MODEL_PATH
opts = Namespace(**opts)
generator_net = pSp(opts)


def apply_projection(latents, vector_path, proj_value):
    """Apply projection to latent vector."""
    vector = np.load(vector_path)
    vector = torch.tensor(vector, dtype=latents.dtype, device=latents.device)
    alpha = proj_value - (torch.sum(latents * vector)
                          / torch.sum(vector * vector))
    edited_latents = latents + (alpha*vector/18)
    if torch.abs(proj_value - torch.sum(edited_latents * vector)
                 / torch.sum(vector * vector)) > 0.1:
        alpha = proj_value - (torch.sum(latents * vector)
                              / torch.sum(vector * vector))
        edited_latents = latents + (alpha*vector/1)
    return edited_latents


def apply_translation(latents, vector_path, scroll_value):
    """Apply translation in latent space."""
    vector = np.load(vector_path)
    vector = torch.tensor(vector, dtype=latents.dtype, device=latents.device)
    latents = latents + scroll_value*vector
    return latents


# Hyper parameters for transformations.
# The stored values are the path to the vectors and the required
# projection value.
LATENT_TRANSFORMATIONS = {
    "Se_0":
    partial(apply_projection,
            vector_path="vectors_editing/custom/sex.npy",
            proj_value=1.2),
    "Se_1":
    partial(apply_projection,
            vector_path="vectors_editing/custom/sex.npy",
            proj_value=-1),
    "Sk_0":
    partial(apply_projection,
            vector_path="vectors_editing/custom/tan.npy",
            proj_value=-1.3),
    "Sk_1":
    partial(apply_projection,
            vector_path="vectors_editing/custom/tan.npy",
            proj_value=-0.7),
    "Sk_2":
    partial(apply_projection,
            vector_path="vectors_editing/custom/tan.npy",
            proj_value=1),
    "Bald":
    partial(apply_projection,
            vector_path="vectors_editing/custom/to_bald.npy",
            proj_value=0.8),
    "make_hair":
    partial(apply_translation,
            vector_path="vectors_editing/custom/from_bald.npy",
            scroll_value=-1.5),
    "A_0":
    partial(apply_projection,
            vector_path="vectors_editing/custom/interface_age.npy",
            proj_value=-30),
    "A_1":
    partial(apply_projection,
            vector_path="vectors_editing/custom/interface_age.npy",
            proj_value=0),
    "A_2":
    partial(apply_projection,
            vector_path="vectors_editing/custom/interface_age.npy",
            proj_value=55),
    "Ch_min":
    partial(apply_projection, vector_path="vectors_editing/custom/chubby.npy",
            proj_value=-0.5),
    "Ch_max":
    partial(apply_projection, vector_path="vectors_editing/custom/chubby.npy",
            proj_value=1.2),
    "Ne_min":
    partial(apply_projection,
            vector_path="vectors_editing/custom/4_7_narrow_eyes.npy",
            proj_value=-5),
    "Ne_max":
    partial(apply_projection,
            vector_path="vectors_editing/custom/4_7_narrow_eyes.npy",
            proj_value=20),
    "B_0":
    partial(apply_projection, vector_path="vectors_editing/custom/bangs.npy",
            proj_value=-1.5),
    "B_1":
    partial(apply_projection, vector_path="vectors_editing/custom/bangs.npy",
            proj_value=3),
    "D_0":
    partial(apply_projection,
            vector_path="vectors_editing/custom/46_4_double_chin.npy",
            proj_value=-0.3),
    "D_1":
    partial(apply_projection,
            vector_path="vectors_editing/custom/46_4_double_chin.npy",
            proj_value=1),
}
IMG_TRANSFORMATIONS = {
    "Hc_0": partial(infer.change_hair_color_smooth, color="black"),
    "Hc_1": partial(infer.change_hair_color_smooth, color="blond"),
    "Hc_2": partial(infer.change_hair_color_smooth, color="blond"),
    "Hc_3": partial(infer.change_hair_color_smooth, color="gray"),
    "Bn_max": infer.make_big_nose,
    "Bp_max": infer.make_big_lips,
    "Be_max": infer.make_bags,
    "Pn_max": infer.make_pointy_nose,
}
IMG_TRANSFORMATIONS_INVERSE = {
    "Bn_min": {
        "function": infer.make_big_nose,
        "value": 1
    },
    "Bp_min": {
        "function": infer.make_big_lips,
        "value": 1
    },
    "Be_min": {
        "function": infer.make_bags,
        "value": 0.5
    },
    "Pn_min": {
        "function": infer.make_pointy_nose,
        "value": 1
    },
}


def encode(img):
    """Encode an image.

    Parameter
    ---------
    img : PIL Image
        Image to encode.

    Returns
    -------
    latents : torch.Tensor
        Encoded image.
    """
    with torch.no_grad():
        latents = generator_net.encoder(
            img_transforms(img).unsqueeze(0).to("cuda").float())[0]
        latents += generator_net.latent_avg
    return latents


def decode(latents):
    """Decode latent vector.

    Parameter
    ---------
    latents : torch.Tensor
        Latent vector to decode.

    Returns
    -------
    edit_image : PIL Image
        Decoded edited image.
    """
    with torch.no_grad():
        edit_image = generator_net.decoder([latents.unsqueeze(0)],
                                           randomize_noise=False,
                                           input_is_latent=True)[0][0]
    edit_image = (edit_image.detach().cpu().transpose(0,
                                                      1).transpose(1,
                                                                   2).numpy())
    edit_image = ((edit_image+1) / 2)
    edit_image[edit_image < 0] = 0
    edit_image[edit_image > 1] = 1
    edit_image = edit_image * 255
    edit_image = edit_image.astype("uint8")
    edit_image = Image.fromarray(edit_image)
    edit_image = edit_image.resize((512, 512))
    return edit_image


def reencode(img):
    """Re-encode a PIL image."""
    latents = encode(img)
    return decode(latents)


def parse_img_name(img_name):
    """Parse image name to extract characteristics."""
    (skin, age, sex, clas, part, bangs, hair_cut, double_chin,
     hair_style) = img_name.split("_")
    hair_style = hair_style.split(".")[0]
    return dict(Sk=skin, A=age, Se=sex, C=clas, P=part, B=bangs, Hc=hair_cut,
                D=double_chin, Hs=hair_style)


POSSIBLE_VALUES = {
    "Sk": {"0", "1", "2"},
    "A": {"0", "1", "2"},
    "Se": {"0", "1"},
    "B": {"0", "1"},
    "Hc": {"0", "1", "2", "3"},
    "D": {"0", "1"},
    "Hs": {"0", "1"}
}
CURSOR_FEATURES = ("Be", "N", "Pn", "Bp", "Bn", "Ch")


def get_img_transformations(img_name, list_of_transformations):
    """Get image transformations from img_name."""
    img_att = parse_img_name(img_name)
    transformations = []
    for att in POSSIBLE_VALUES:
        for val in POSSIBLE_VALUES[att]:
            if img_att[att] != val:
                t_name = att+"_"+val
                if t_name in list_of_transformations:
                    transformations.append(t_name)
    for att in CURSOR_FEATURES:
        for t_name in [att+"_max", att+"_min"]:
            if t_name in list_of_transformations:
                transformations.append(t_name)
    if "Bald" in list_of_transformations and img_att["Hc"] != 4:
        transformations.append("Bald")
    return transformations


def process_img(img_path, output_path, list_of_transformations):
    """Process an image and save it to output_path."""
    img_name = os.path.basename(img_path)

    output_path = os.path.join(output_path, img_name[:-4])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    img_att = parse_img_name(img_name)
    img_transforms = get_img_transformations(img_name, list_of_transformations)

    img = Image.open(img_path).resize((512, 512))
    parsing = infer.compute_mask(img, parsing_net)
    latents = encode(img)

    for transfo in img_transforms:
        if transfo in LATENT_TRANSFORMATIONS:
            edited_img = decode(LATENT_TRANSFORMATIONS[transfo](latents))
            edited_img.save(os.path.join(output_path, transfo + ".png"))

    for transfo in img_transforms:
        if transfo in IMG_TRANSFORMATIONS:
            if "Hc" in transfo and img_att["Hc"] == "4":
                edited_img = decode(
                    LATENT_TRANSFORMATIONS["make_hair"](latents))
                edited_img = IMG_TRANSFORMATIONS[transfo](np.array(edited_img),
                                                          parsing)
            else:
                edited_img = IMG_TRANSFORMATIONS[transfo](np.array(img),
                                                          parsing)
            edited_img = reencode(Image.fromarray(edited_img))
            edited_img.save(os.path.join(output_path, transfo + ".png"))

    for transfo in img_transforms:
        if transfo in IMG_TRANSFORMATIONS_INVERSE:
            edited_img = IMG_TRANSFORMATIONS_INVERSE[transfo]["function"](
                np.array(img), parsing)
            edited_latents = encode(Image.fromarray(edited_img))
            edited_img = decode(latents +
                                (IMG_TRANSFORMATIONS_INVERSE[transfo]["value"]
                                 * (latents-edited_latents)))
            edited_img.save(os.path.join(output_path, transfo + ".png"))


def apply_e4e(data_dir, output_path, configs):
    """Process e4e in all images in input dataset.

    Parameters
    ----------
    data_dir : str
        Path to the dataset folder.
    output_path : str
        Path to the output folder.
    configs : dict
        Contains the list of transformations to apply to images
        (key 'transformations').
    """
    list_of_transformations = configs['transformations']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n_images = len(os.listdir(data_dir))
    for i, img_name in enumerate(os.listdir(data_dir)):
        img_path = os.path.join(data_dir, img_name)
        process_img(img_path, output_path, list_of_transformations)
        print(f'image {i + 1}/{n_images} done', end='\r')
    print()


if __name__ == "__main__":
    print('Apply encoder4editing...')
    DATA_DIR = "data/face_challenge"
    OUTPUT_PATH = "res/run1/output_images"
    CONFIGS = {'transformations': ["A_0", "B_0", "Bald", "Bn_min", "Bp_max", "Ch_min", "D_0",
                                   "Hc_3", "Pn_min", "Se_1"]}
    apply_e4e(DATA_DIR, OUTPUT_PATH, CONFIGS)
