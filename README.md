# Face editing with Style GAN 2 and facial segmentation (Ceteris Paribus Face Challenge Intercentrales 2022)

[![Release](https://img.shields.io/github/v/release/valentingol/gan-face-editing)](https://github.com/valentingol/gan-face-editing/releases)
![PythonVersion](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)
[![License](https://img.shields.io/github/license/valentingol/gan-face-editing?color=brightgreen)](https://stringfixer.com/fr/MIT_license)

[![Pycodestyle](https://github.com/valentingol/gan-face-editing/actions/workflows/pycodestyle.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/pycodestyle.yaml)
[![Flake8](https://github.com/valentingol/gan-face-editing/actions/workflows/flake.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/flake.yaml)
[![Pydocstyle](https://github.com/valentingol/gan-face-editing/actions/workflows/pydocstyle.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/pydocstyle.yaml)
[![Isort](https://github.com/valentingol/gan-face-editing/actions/workflows/isort.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/isort.yaml)
[![PyLint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/valentingol/c60e6ce49447254be085193c99b8425b/raw/gan_face_editing_pylint_badge.json)](https://github.com/valentingol/gan-face-editing/actions/workflows/pylint.yaml)

Winner team repository of Inter-Centrales 2022 AI competition: Ceteris Paribus Face Challenge: [site of the competition](https://transfer-learning.org/competition.html).

This work is under the MIT license.

![alt text](ressources/images/compet_img.png)

This repository uses third-party works:

- [anycost-gan](https://github.com/mit-han-lab/anycost-gan) (MIT license)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (MIT license)
- [DensePredictionTransformer / DPT](https://github.com/isl-org/DPT) (MIT license)

Licenses are provided in `third_party_licenses`.

## To-Do list

- [x] Project images in latent space
- [x] Modify bounds of the direction
- [x] Find translations in latent space
- [x] Saving pipeline for direction
- [x] Solve bald issue
- [x] Detect and improve bad translations
- [x] Solve specific issues (manually or with other methods)
- [x] Focused change by semantic segmentation
- [x] Resolve some artifacts and background problems with depth estimation
- [x] Refactor the code to make it more convenient
- [x] Add a convenient config system
- [ ] Improve realism with GFP GAN - IN PROGRESS 🚧
- [ ] Look for other repo to solve skin and age
- [ ] Test GAN retraining and extract direction features

## Quick Start

**Note**: you need a Nvidia GPU to run the processing. Only editor API with pre-computed latent vectors is available with CPU.

### Installation

Clone this repository and pull the models required from postprocessing via LFS:

```script
git clone git@github.com:valentingol/gan-face-editing.git
cd gan-face-editing
git lfs pull
```

Then, create a new virtual environment and install all the required packages:

```bash
pip install -e .
pip install -r requirements.txt
```

*NOTE*: to run on multiple GPUs, you should also install [horovod](https://github.com/horovod/horovod) (Mac or Linux only).

The original dataset of the competition is available here: [drive dataset](https://drive.google.com/drive/folders/1-R1863MV8CuCjmycsLy05Uc6bdkWfuOP?usp=sharing)

Unzip the content in a folder called `data/face_challenge`.

By default, a depth estimation is performed to correct artifacts of the backgrounds. You need download the model for depth estimation here: [download](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view) and put it on `postprocess/depth_segmentation/model/`.

### Compute latent space of images

Once the data are downloaded, you must compute the projected latent vectors of the images. It can take some time to compute as the script optimize the latent vector through multiple gradient descent steps but you can significantly reduce the time by reducing the number of iterations in configurations (0 iteration mean that you get the latent vector computed by the pretrained encoder). By default, it is 200 iterations.

```bash
python apps/project_images.py [--projection.n_iter=<n_iter>]
```

### Editor API

Optionally, can run following the script to launch the image editing API.

![Alt Text](ressources/gif/editor_api.gif)

```bash
# if you have a Nvidia GPU:
FORCE_NATIVE=1 python editor_API.py
# otherwise:
python apps/editor.py
```

In this API you can visualize and edit the reconstructed images using attribute cursors to build the changes you want. Default translations are already available in this repository but you can edit your own with the button "Save trans". The translations will be saved at `projection/run1/translations_vect`. You can also save the edited images with the button "Save img". The images will be saved in `projection/run1/images_manual_edited`. You can have more information about creating your own translation in the "Save your own translations" section. **Note that this repository already provide a lot of preconstructed translations.**

### Translation and postprocessing pipeline

You can now run the full pipeline to apply automatic translation on all the input images and apply three steps of postprocessing (in this order):

- domain mixup (using constant area of interest)
- segmentation mixup (using ResNet)
- depth segmentation mixup (using ViT)

```bash
FORCE_NATIVE=1 python apps/run_pipeline.py
```

All steps of the pipeline can be run individually and the results after all steps are saved in `res/run1`.

The final images are saved under `res/run1/images_post_depth_segmentation`.

You can avoid using the pre-computed translation with `--translation.use_precomputed=False`.

## Postprocessing

All transformation can be run individually even if the project was designed to work with `apps/run_pipeline.py` in order to be consistent with custom configurations (see "Configurations" section).

### Domain Mixup

To keep only the modification near the area where we expect them (called 'domain') for a specific change, a mixup is applied. The idea is to draw an area in a 512x512 white image that correspond to the area we expect add the changes, for each characteristic. The images are saved in `postprocess/domain_mixup/images/` (black and white images and black pixels correspond to the domain).

```bash
python pipeline/domain_mixup.py
```

By default, the resulting images are in `res/run1/images_post_domain_mixup/` and the distances from all pixels to the domains are saved in `postprocess/domain_mixup/distances`.

As domain mixup shows no significant improvement and can even add some unwanted artifacts, it is disable by default. You can enable it with `--pipeline.skip_domain_mixup=False`.

### GFP-GAN

To enhance the quality of the areas of the face modified, you can incorporate a read-to-use GAN, GFP-GAN. It was specifically trained to restore the quality of ancient images of faces, and is open-sourced : https://github.com/TencentARC/GFPGAN.

However, it only works well on faces, so you might want to use segmentation and use GFP-GAN only on the face part, or even the mask of the area of modifications.

In order to use it, you first have to download a trained model, at https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth, and store it, eg in `models/gpf_gan`. The code that calls it and run this part of the pipeline is in `pipeline/gpf_gan.py`. You will have to specify the paths of the input images, the output images and the model. 

### Segmentation

To go further in the last idea, we apply a semantic segmentation on the original image and the edited images in order to find for each image and for each transformation the area where we expect to find the change. First you need to get the segmentation model by Git LFS (see [Git LFS](https://git-lfs.github.com/) for details):

```bash
git lfs pull
```

Note than you can use an other model you want as long as it is compatible with the models used by [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). You should modify the configurations to set the path of your new model.

Then you can run the following script to merge the previously edited images with the original ones by semantic segmentation:

```bash
python pipeline/segment.py
```

By default, the resulting images are in `res/run1/images_post_segmentation/`

### Depth estimation

In order to improve the segmentation, we perform depth estimation to correct artifacts of the backgrounds for the images coming from the segmentation. First, you need get the model for depth estimation. You can download it here: [download](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view) and put it on `postprocess/depth_segmentation/model/`.
The idea is to make a depth estimation with a transformer model (see [DPT](https://github.com/isl-org/DPT)). Then we build the foreground mask with a K-means algorithm. This allows to extract a relevant foreground from the segmented image and to paste it on the background of the original image.

Then you can run the following script to merge the previously edited images with the original ones by depth estimation:

```bash
python pipeline/depth_segmentation.py
```

By default, the resulting images are in `res/run1/images_post_depth_segmentation/`

## To go further

### Configuration

This project use the [rr-ml-config](https://gitlab.com/reactivereality/public/rr-ml-config-public) configuration system to configure all the main functions (in `apps/`). You can find all the configurations on yaml files in the `config/` folder. The default configs are in `config/default/` folder and split on multiple files (creating multiple sub-configs) to make it easier to use. Then, you can create your own experiments by editing a file `config/exp/my_exp.yaml` and add lines to overwriting some default configs (an example is provided in `config/exp/base.yaml`, that is used by default). Then you can run the `apps/` functions using your experiment configs. For instance:

```bash
python apps/project_images.py --config config/exp/my_exp.yaml
FORCE_NATIVE=1 python apps/editor.py --config config/exp/my_exp.yaml
FORCE_NATIVE=1 python apps/run_pipeline.py --config config/exp/my_exp.yaml
```

You can also change all the configurations with command line arguments and combine the two. For instance:

```bash
python apps/project_images.py --config config/exp/my_exp.yaml --projection.n_iter=50 --projection.enc_reg_weight=0.5
FORCE_NATIVE=1 python apps/editor.py --config config/exp/my_exp.yaml --editor.n_style_to_change=8
FORCE_NATIVE=1 python apps/run_pipeline.py --config config/exp/my_exp.yaml --segmentation.margin=10
```

This make your experiments very convenient because you can set your main configurations in your configuration file and you no longer need to write all your configurations in the command line.

All your configurations are saved that allow full reproducibility of your experiments.

Each time you want to create a new experiments configuration, you need to overwrite the projection dir with the name of your "projection run" (e.g. `projection/run_with_1000_iter`), the result dir (with the name of your "pipeline run" (e.g. `res/test_with_domain_mixup`), and the save path with the name of your "global run", (e.g. `configs/runs/1000_iter_proj_with_domain_mixup`). By default, it is "run1", "run1" and "run1&1" respectively.

### Save your own translations

To save the translations (= latent direction) you want in the `projection/run1` folder, you can click on the "Save trans" button and set the name of the direction. In all cases, the pipeline will try to use your new translation to modify the images.

You can use this project for two purposes:

- apply automatically some changes in all the images with the same manner (by default)
- use already known characteristics of the images to adapt the changes (use `--translation.use_caracs_in_img=True` when running the pipeline or edit you configuration file, see "Configurations" section for details)

**In this second case, the images should have a specific name indicating the characteristics of the image. The name should follow the rule presented [here](https://transfer-learning.org/rules). Moreover, the translations vector should also have a specific name.**

- for 'cursor' transformation (min - max): `{carac}_{'min' or 'max'}`. Example `Be_min` means cursor transformation to minimal bags under the eyes (Be_min).

- for default transformation (transformation that doesn't depend on the current value of the characteristic): `{carac}_{new_value}`. Example: `Hc_0` means default transformation to black hair color (Hc_0)

- for specific transformation (transformation that depends on the current value of the characteristic) that will overwrite default transformation: `{carac}_{new_value}_fr_{old_value}`. Example `A_0_fr_1` means specific transformation from intermediate age (A_1) to young age (A_0), 'fr' means 'from'.

All required transformations must be treated for all input characteristics to make a valid submission. Note that this repository contains enough default translations to make a submission.

Now, given an input image, the transformations to create characteristics that are not in the initial image can automatically be processed with the function `pipeline/utils/translation/get_translations.py`.

Note that if you not used the characteristics in the image, you can name the translation vectors as you want.

### Extract other attribute directions

:construction:

### Retrain the GAN

To retrain the GAN you need to install [horovod](https://github.com/horovod/horovod) (Mac or Linux only).

:construction:

### Align faces

This repository provides a function to align faces. It requires [dlib](https://pypi.org/project/dlib/).

:construction:
