# Code from https://github.com/mit-han-lab/anycost-gan
# Code adapted from https://gist.github.com/lzhbrian/
# bde87ab23b499dd02ba4f588258f57d5
"""Face alignment with FFHQ method.

link: https://github.com/NVlabs/ffhq-dataset
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark models from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import argparse
import os
import sys

import dlib  # pip install dlib if not found
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage

# download models from: http://dlib.net/files/shape_predictor_68_face_
# landmarks.dat.bz2
predictor = dlib.shape_predictor('anycostgan/shape_predictor/'
                                 'shape_predictor_68_face_landmarks.dat')


def get_landmark(filepath):
    """Get landmark with dlib.

    Parameters
    ----------
    filepath : str
        Path to image.

    Returns
    -------
    np.array
        Landmarks, shape=(68, 2).
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    if len(dets) == 0:
        print(' * No face detected.')
        sys.exit()

    # my editing here
    if len(dets) > 1:
        print(f' * WARNING: {len(dets)} faces detected in the image. '
              'Only preserve the largest face.')
        img_ws = [d.right() - d.left() for d in dets]
        largest_idx = np.argmax(img_ws)
        dets = [dets[largest_idx]]

    assert len(dets) == 1
    shape = predictor(img, dets[0])

    parts = list(shape.parts())
    landmarks_list = []
    for part in parts:
        landmarks_list.append([part.x, part.y])
    landmarks = np.array(landmarks_list)
    return landmarks


def align_face(filepath):
    """Align face.

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    PIL Image:
        Aligned face image.
    """
    landmark = get_landmark(filepath)

    lm_eye_left = landmark[36:42]  # left-clockwise
    lm_eye_right = landmark[42:48]  # left-clockwise
    lm_mouth_outer = landmark[48:60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left+eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    eye_to_mouth = (mouth_left+mouth_right) * 0.5 - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    center = eye_avg + eye_to_mouth*0.1
    quad = np.stack(
        [center - x - y, center - x + y, center + x + y, center + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(
                np.float32(x) / pad[0],
                np.float32(w - 1 - x) / pad[2]), 1.0 - np.minimum(
                    np.float32(y) / pad[1],
                    np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0])
                - img) * np.clip(mask*3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Align face images to FFHQ format")
    parser.add_argument("-i", "--input", type=str,
                        help="path of the input image")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output path of the aligned face")

    args = parser.parse_args()

    img = align_face(args.input)
    if args.output is None:
        n1, n2 = os.path.splitext(args.input)
        args.output = n1 + '-aligned' + n2
    print(f' * Saving the aligned image to {args.output}')
    img.save(args.output)
