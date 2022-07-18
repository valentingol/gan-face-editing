# Face editing with Style GAN 2 and facial segmentation (Ceteris Paribus Face Challenge Intercentrales 2022)

[![Release](https://img.shields.io/github/v/release/valentingol/gan-face-editing)](https://github.com/valentingol/gan-face-editing/releases)
![PythonVersion](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)
[![License](https://img.shields.io/github/license/valentingol/gan-face-editing?color=brightgreen)](https://stringfixer.com/fr/MIT_license)

[![Pycodestyle](https://github.com/valentingol/gan-face-editing/actions/workflows/pycodestyle.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/pycodestyle.yaml)
[![Flake8](https://github.com/valentingol/gan-face-editing/actions/workflows/flake.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/flake.yaml)
[![Pydocstyle](https://github.com/valentingol/gan-face-editing/actions/workflows/pydocstyle.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/pydocstyle.yaml)
[![Isort](https://github.com/valentingol/gan-face-editing/actions/workflows/isort.yaml/badge.svg)](https://github.com/valentingol/gan-face-editing/actions/workflows/isort.yaml)
[![PyLint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/valentingol/c60e6ce49447254be085193c99b8425b/raw/gan_face_editing_pylint_badge.json)](https://github.com/valentingol/gan-face-editing/actions/workflows/pylint.yaml)

**NOTE** : This is work based on the winner team repository of Inter-Centrales 2022 AI competition: Ceteris Paribus Face Challenge: [site of the competition](https://transfer-learning.org/competition.html) plus the work of the second team composed by Thibault Le Sellier De Chezelles and Hédi Razgallah (original work can be found [here](https://github.com/HediRaz/InterCentrales).

This repository and the repositories it contains are licensed under the [MIT license](LICENSE.md).

---

![alt text](ressources/images/compet_img.png)

This repository uses third-party works:

- [anycost-gan](https://github.com/mit-han-lab/anycost-gan) (MIT license)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (MIT license)
- [DensePredictionTransformer / DPT](https://github.com/isl-org/DPT) (MIT license)
- [encoder4editing](https://github.com/omertov/encoder4editing)

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
- [x] Improve realism with GFP GAN
- [x] Look for other repo to solve skin and age in data_challenge branch
- [ ] Fix artifact bugs (IN PROGRESS 🚧)
- [ ] Allow using GFP GAN and depth estimation on a subset of transformations/images
- [ ] Improve depth estimation speed with a lighter model
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

Once the data are downloaded, you must compute the projected latent vectors of the images. It can take some time to compute as the script optimize the latent vector through multiple gradient descent steps but you can significantly reduce the time by reducing the number of iterations in configurations (0 iteration mean that you get the latent vector computed by the pre trained encoder). By default, it is 200 iterations.

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

- encoder4editing GAN
- GFP GAN
- segmentation mixup (using ResNet)
- depth segmentation mixup (using ViT)

```bash
FORCE_NATIVE=1 python apps/run_pipeline.py
```

All steps of the pipeline can be run individually and the results after all steps are saved in `res/run1`.

You can avoid using the pre-computed translation with `--translation.use_precomputed=False`. Then, the only translations that will be used are the ones you have created with the editor API.

## Postprocessing

The pipeline of transformations include many transformations. All of them can be run individually even if the project was designed to work with `apps/run_pipeline.py` in order to be consistent with custom configurations (see "Configurations" section).

## encoder4editing

**IMPORTANT**: To run encoder4editing you must download the pre trained model here: [e4e_ffhq_encode.pt]((https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)) and put in in `postprocess/encoder4editing/model/e4e_ffhq_encode.pt`.

 The encoder4editing (e4e) encoder is specifically designed to complement existing image manipulation techniques performed over StyleGAN's latent space. It is based on the following paper: [Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766). It is used instead of AnycostGAN for some transformation if needed. By default, no transformation are made with this encoder.

 You can run individually this step with:

 ```script
python pipeline.encoder4editing.py
 ```

### GFP-GAN

To enhance the quality of the areas of the face modified, you can incorporate a ready-to-use GAN, GFP-GAN. It was specifically trained to restore the quality of ancient images of faces, and is open-sourced : [GFP GAN](https://github.com/TencentARC/GFPGAN).

However, it only works well on faces, so you might want to use segmentation and use GFP-GAN only on the face part, or even the mask of the area of modifications.

In order to use it, you first have to download a trained model, with `git lfs pull` or following this [link](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth), and store it, in `postprocess/gpf_gan/model`. The code that calls it and runs this part of the pipeline is in `pipeline/gpf_gan.py`.

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

By default, the resulting images are in `res/run1/output_images`

**Imortant**: The translations need to have a valid prefix in their name to apply the segmentation. See the table in the section *Save your own translations* for more information.

### Depth estimation

In order to improve the segmentation, we perform depth estimation to correct artifacts of the backgrounds for the images coming from the segmentation. First, you need get the model for depth estimation. You can download it here: [download](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view) and put it on `postprocess/depth_segmentation/model/`.
The idea is to make a depth estimation with a transformer model (see [DPT](https://github.com/isl-org/DPT)). Then we build the foreground mask with a K-means algorithm. This allows to extract a relevant foreground from the segmented image and to paste it on the background of the original image.

Then you can run the following script to merge the previously edited images with the original ones by depth estimation:

```bash
python pipeline/depth_segmentation.py
```

By default, the resulting images are in `res/run1/output_images`

## To go further

### Configuration

This project use the [rr-ml-config](https://gitlab.com/reactivereality/public/rr-ml-config-public) configuration system to configure all the main functions (in `apps/`). You can find all the configurations on yaml files in the `config/` folder. The default configs are in `config/default/` folder and split on multiple files (creating multiple sub-configs) to make it easier to use. Then, you can create your own experiments by editing a file `config/exp/my_exp.yaml` and add lines to overwriting some default configs (an example is provided in `config/exp/base.yaml`, that is used by default). Then you can run the `apps/` functions using your experiment configs. For instance:

```bash
FORCE_NATIVE=1 python apps/project_images.py --config config/exp/my_exp.yaml
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

### Modify your own images

You can modify your own dataset of images by create a new folder `data/my_dataset_name` and add it in your configs:

```yaml
# configs/exp/
data_dir: data/my_dataset_name
```

Then, you need to align the faces on your images following the [FFHQ alignement](https://www.kaggle.com/datasets/arnaud58/ffhq-flickr-faces-align-crop-and-segment) and resize them to 512 * 512 pixels. First you need to install [dlib](https://pypi.org/project/dlib/) and downoad the [shape predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place it in `anycostgan/shape_predictor/shape_predictor_68_face_landmarks.dat`. Then you can run the following script. Note that data will be transformed in `data_dir` **in place** so save a backup if you want before.

```bash
python apps/preprocess_images.py --config configs/exp/<exp_config_file>.yaml
```

Now you can project the images, use the editor and run your pipeline of transformations such as the section above.

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

**Important:** To use the semantic segmentation mixup, the algorithm should understand the part of the face you want to modify (and the part of the image you want to preserve from the original image). To do so, you need to add a particular prefix for the translation vectors: for instance for eyes change, you need to use the prefix 'N'. For instance `N_0` is a valid translation that will only edit the eyes. The table of prefix is:

| Part to modify    | Prefix             |
| :---------------: |:------------------:|
| All the face      | A, Ch, Se or Sk    |
| Hair              | B, Hc or Hs        |
| Nose              | Pn                 |
| Eyes              | Bn or N            |
| Just under eyes   | Be                 |
| Lips              | Bp                 |

If you don't use one of the prefix above, no segmentation mixup will be applied. You can use the prefix you want for custom transformations. More intuitive prefix will be available later (e.g 'eyes' for eyes, 'face' for all the face ...).

### Extract other attribute directions

:construction:

### Retrain the GAN

To retrain the GAN you need to install [horovod](https://github.com/horovod/horovod) (Mac or Linux only).

:construction:
