"""Inference utils for face segmentation."""

import os.path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pipeline.utils.segmentation.model import BiSeNet

matplotlib.use("TkAgg")


def load_model(checkpoint='79999_iter.pth'):
    """Load face parsing model."""
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('postprocess/segmentation/model', checkpoint)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net


def vis_parsing_maps(img, parsing_anno):
    """Visualize parsing annotation on the image."""
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85],
                   [255, 0, 170], [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255],
                   [170, 0, 255], [0, 85, 255], [0, 170, 255], [255, 255, 0],
                   [255, 255, 85], [255, 255, 170], [255, 0, 255],
                   [255, 85, 255], [255, 170, 255], [0, 255, 255],
                   [85, 255, 255], [170, 255, 255]]

    img = np.array(img)
    vis_im = img.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for clas in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == clas)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[clas]
    index = np.where(vis_parsing_anno == 0)
    vis_parsing_anno_color[index[0], index[1], :] = 0

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = 0.4*vis_im + 0.6*vis_parsing_anno_color
    vis_im = np.uint8(vis_im)

    plt.imshow(vis_im)
    plt.show()


def compute_mask(img, net):
    """Compute the face mask of the image.

    Parameters
    ----------
    img : PIL.Image
        Input image to be segmented.
    net : torch.nn.Module
        The model to use for segmentation.

    Returns
    -------
    parsing : np.ndarray
        Array of the mask.
    """
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    return parsing


def get_background_mask(parsing, add_clothes=False, add_glasses=False,
                        add_earings=False):
    """Get the background mask of the image."""
    mask = np.logical_or(parsing == 0, parsing == 18)
    if add_clothes:
        mask = np.logical_or(mask, parsing == 16)
    if add_glasses:
        mask = np.logical_or(mask, parsing == 6)
    if add_earings:
        mask = np.logical_or(mask, parsing == 9)
    indexes = np.where(mask)
    g_i = int(np.mean(indexes[0]))
    g_j = int(np.mean(indexes[1]))
    return mask, (g_i, g_j)


def get_nose_mask(parsing):
    """Get the nose mask of the image."""
    mask = parsing == 10
    return mask


def get_left_eye_mask(parsing):
    """Get the left eye mask of the image."""
    mask = parsing == 4
    return mask


def get_right_eye_mask(parsing):
    """Get the right eye mask of the image."""
    mask = parsing == 5
    return mask


def get_mouth_mask(parsing):
    """Get the mouth mask of the image."""
    mask = np.logical_or(parsing == 11, parsing == 12)
    mask = np.logical_or(mask, parsing == 13)
    return mask


def get_hair_mask(parsing):
    """Get the hair mask of the image."""
    mask = parsing == 17
    return mask


def delete_foreground(img, foreground_mask, delta=5.0):
    """Delete the foreground of the image.

    Parameters
    ----------
    img : PIL.Image
        Input image to be segmented.
    foreground_mask : jnp.ndarray
        Array of the foreground mask to delete.
    delta : float, optional
        The distance to the foreground to delete. By default, 5.0.

    Returns
    -------
    img : PIL.Image
        The image with the foreground deleted.
    """
    f_indexes = np.where(foreground_mask)
    img[f_indexes[0], f_indexes[1], :] = 0

    for i in np.unique(f_indexes[0]):
        arg_i = f_indexes[0] == i
        indexes_j = f_indexes[1][arg_i]
        j_max = max(indexes_j)
        j_min = min(indexes_j)
        j_max = min(511, j_max + delta)
        j_min = max(0, j_min - delta)
        img[i, indexes_j[indexes_j < (j_min+j_max) / 2], :] = img[i, j_min]
        img[i, indexes_j[indexes_j >= (j_min+j_max) / 2], :] = img[i, j_max]

    img = np.where(
        np.stack((foreground_mask, foreground_mask, foreground_mask), -1),
        cv2.medianBlur(img, 5), img)
    return img


def replace_obj(origin_image, edit_image, origin_mask, edit_mask, o_pos, e_pos,
                background_mask=None, delta=5, blur_foreground=False):
    """Replace the object in the image."""
    if blur_foreground:
        origin_image = delete_foreground(origin_image, origin_mask,
                                         delta=delta)

    e_indexes = np.where(edit_mask)
    o_indexes = [
        e_indexes[0] + (o_pos[0] - e_pos[0]), e_indexes[1] +
        (o_pos[1] - e_pos[1]),
    ]
    o_indexes[0] = np.clip(o_indexes[0], 0, 511)
    o_indexes[1] = np.clip(o_indexes[1], 0, 511)
    x, y = o_indexes[0], o_indexes[1]
    if background_mask is not None:
        origin_bg = origin_image[background_mask].copy()
        origin_image[x, y, :] = edit_image[x, y, :]
        origin_image[background_mask] = origin_bg
    else:
        origin_image[x, y, :] = edit_image[x, y, :]

    return origin_image


def change_hair_color_brut(img, parsing, color="blond"):
    """Change the hair color of the image.

    Replace original hair with new hair with brut transformation.
    """
    hair_mask = get_hair_mask(parsing)

    if color == "blond":
        color_a = np.array(Image.open("face_parsing/makeup/blond_hair.png"))
        color_a = color_a[:512, :512]
    if color == "brown":
        color_a = np.array(Image.open("face_parsing/makeup/brown_hair.png"))
        color_a = color_a[:512, :512]
    if color == "black":
        color_a = np.array(Image.open("face_parsing/makeup/black_hair.png"))
        color_a = color_a[:512, :512]
    if color == "gray":
        color_a = np.array(Image.open("face_parsing/makeup/gray_hair.png"))
        color_a = color_a[:512, :512]

    edit_img = img.copy()
    edit_img[hair_mask] = color_a[hair_mask]
    return edit_img


def change_hair_color_smooth(img, parsing, color="blond"):
    """Change the hair color of the image.

    Replace original hair with new hair with smooth transformation.
    """
    hair_mask = get_hair_mask(parsing)

    edit_img = img.copy()
    if color == "blond":
        color_a = np.array([245, 232, 39])
        edit_img[hair_mask] = 0.3*color_a + 0.7 * edit_img[hair_mask]
    if color == "brown":
        color_a = np.array([88, 51, 3])
        edit_img[hair_mask] = 0.6*color_a + 0.4 * edit_img[hair_mask]
    if color == "black":
        color_a = np.array([0, 0, 0])
        edit_img[hair_mask] = 0.8*color_a + 0.2 * edit_img[hair_mask]
    if color == "gray":
        color_a = np.array([200, 200, 200])
        edit_img[hair_mask] = 0.6*color_a + 0.4 * edit_img[hair_mask]

    return edit_img


def make_shape_under_eye(img, mask_eye, color: list, maxi=True):
    """Make a large shape under mask_eye, of the color 'color'.

    maxi=True means we keep the color, else we average the color under
    the mask and add 60 to lighten it further.
    """
    coords = np.stack(np.where(mask_eye), axis=-1)
    d_x = np.max(coords[:, 1]) - np.min(coords[:, 1])
    d_y = np.max(coords[:, 0]) - np.min(coords[:, 0])
    coords[:, 0] += d_y
    if not maxi:
        color = np.clip(
            np.mean(img[coords[:, 0], coords[:, 1], :], axis=0) + 60, 0, 255)
    for x in range(-1, 2):
        for y in range(0, 3):
            tmp_coords = np.copy(coords)
            tmp_coords[:, 1] += x * d_x // 2
            tmp_coords[:, 0] += y * d_y // 2
            img[tmp_coords[:, 0], tmp_coords[:, 1], :] = np.array(color)
    return img


def make_bags(img: np.array, parsing, maxi=True, occident=True):
    """Add drawings for bags under eyes feature under the eye.

    maxi=True means we darken the eye, lighten if False
    occident=True means clear skin color, False means dark
    """
    if occident:
        color = [120, 81, 100]
    else:
        color = [60, 40, 50]
    img = make_shape_under_eye(img, get_left_eye_mask(parsing), color,
                               maxi=maxi)
    img = make_shape_under_eye(img, get_right_eye_mask(parsing), color,
                               maxi=maxi)
    return img


def replace(o_img, e_img, mask):
    """Put the pixels in e_img described by mask into o_img."""
    indexes = np.where(mask)
    img = np.copy(o_img)
    img[indexes[0], indexes[1], :] = e_img[indexes[0], indexes[1], :]
    return img


def replace_background(o_img, e_img, net):
    """Compute the background mask of o_img.

    Compute the background mask of o_img, replaces the foreground by
    averaging. Then takes e_img's foreground and puts it into o_img
    """
    o_parsing = np.array(compute_mask(o_img, net))
    e_parsing = np.array(compute_mask(e_img, net))

    o_truth = [o_parsing == i for i in list(range(1, 16)) + [17]]
    o_mask = np.zeros_like(o_truth[0], dtype=bool)
    for i in o_truth:
        o_mask += i

    img = delete_foreground(np.array(o_img), o_mask)

    e_truth = [e_parsing == i for i in list(range(1, 16)) + [17]]
    e_mask = np.zeros_like(e_truth[0], dtype=bool)
    for i in e_truth:
        e_mask += i

    img = replace(img, np.array(e_img), e_mask)

    return img


def make_pointy_nose(img, parsing):
    """Add a nose colored bar on the nose.

    That's 20% longer than the nose and half of its width.
    """
    mask = get_nose_mask(parsing)

    indexes = list(np.where(mask))
    color = np.clip(
        np.mean(img[indexes[0], indexes[1], :], axis=0) - 0, 0, 255)
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    y_min = np.min(indexes[0])
    y_max = np.max(indexes[0])
    d_x = x_max - x_min
    d_y = y_max - y_min

    x_bounds = [x_min + d_x//4, x_max - d_x//4]

    cond = np.logical_and(indexes[1] > x_bounds[0], indexes[1] < x_bounds[1])

    indexes[0] = indexes[0][cond]
    indexes[1] = indexes[1][cond]

    img[indexes[0], indexes[1], :] = color

    indexes[0] += d_y // 5
    img[indexes[0], indexes[1], :] = color

    return img


def make_flat_nose(img, parsing):
    """Add a nose colored bar on the nose.

    That's 20% longer than the nose and half of its width.
    """
    mask = get_nose_mask(parsing)
    img = make_pointy_nose(img, parsing)
    indexes = list(np.where(mask))
    color = np.array([0, 0, 0])
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    y_min = np.min(indexes[0])
    y_max = np.max(indexes[0])
    d_x = x_max - x_min
    d_y = y_max - y_min

    y_bounds = [y_min + 3*d_y//4, y_max]

    cond = np.logical_and(indexes[0] > y_bounds[0], indexes[0] < y_bounds[1])

    indexes[0] = indexes[0][cond]
    indexes[1] = indexes[1][cond]

    indexes[0] += d_y // 10

    img[indexes[0], indexes[1], :] = color

    indexes[1] += d_x // 5
    img[indexes[0], indexes[1], :] = color

    indexes[1] -= 2 * d_x // 5
    img[indexes[0], indexes[1], :] = color

    return img


def make_balls_around_mouth(img, parsing):
    """Make balls around the mouth."""
    mask_mouth = get_mouth_mask(parsing)

    indexes = np.where(mask_mouth)
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    y_min = np.min(indexes[0])
    d_x = int((x_max-x_min) * 1.3)

    square_indexes = [
        np.concatenate([np.arange(0, d_x) for _ in range(d_x)], axis=0),
        np.concatenate([np.ones(d_x) * i for i in range(d_x)], axis=0)
    ]

    coords1 = [
        square_indexes[0] + y_min - d_x//2,
        (square_indexes[1] + x_min - d_x).astype(int)
    ]
    coords2 = [
        square_indexes[0] + y_min - d_x//2,
        (square_indexes[1] + x_min + d_x).astype(int)
    ]

    print(coords1, coords2)

    color = (np.clip(np.mean(img[coords1[0], coords1[1], :], axis=0), 0, 255)
             + np.clip(np.mean(img[coords2[0], coords2[1], :], axis=0), 0,
                       255)) / 2 + 25

    img[coords1[0], coords1[1], :] = color
    img[coords2[0], coords2[1], :] = color

    return img


def make_bigger(img, mask):
    """Make the face bigger."""
    img = np.array(img)
    idx, idy = np.where(mask)
    try:
        minx, maxx = min(idx), max(idx)
        miny, maxy = min(idy), max(idy)
    except ValueError:  # mask empty
        return img
    g_x, g_y = (minx+maxx) // 2, (miny+maxy) // 2
    mask = np.float32(mask[minx:maxx, miny:maxy])
    new_shape = (int(mask.shape[0] * 1.5), int(mask.shape[1] * 1.5))
    mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_CUBIC)
    mask = mask > 0.3
    mask = np.stack((mask, mask, mask), -1)
    obj = np.uint8(
        cv2.resize(img[minx:maxx, miny:maxy], new_shape,
                   interpolation=cv2.INTER_CUBIC))
    minx, maxx = (g_x - new_shape[1] // 2,
                  g_x + new_shape[1] // 2 + new_shape[1] % 2)
    miny, maxy = (g_y - new_shape[0] // 2,
                  g_y + new_shape[0] // 2 + new_shape[0] % 2)
    img[minx:maxx, miny:maxy] = np.where(mask, obj, img[minx:maxx, miny:maxy])
    return img


def make_big_nose(img, parsing):
    """Make the nose bigger."""
    nose_mask = get_nose_mask(parsing)
    img = make_bigger(img, nose_mask)
    return img


def make_big_lips(img, parsing):
    """Make the lips bigger."""
    lips_mask = get_mouth_mask(parsing)
    img = make_bigger(img, lips_mask)
    return img
