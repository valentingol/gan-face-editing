# Face editing with Style GAN 2 and facial segmentation (Ceteris Paribus Face Challenge Intercentrales 2022)

Repository of Inter-Centrales 2022 AI competition: Ceteris Paribus Face Challenge: [site of the competition](https://transfer-learning.org/competition.html).

![alt text](assets/figures/compet_img.png)

## To-Do list

- [x] Project images in latent space
- [x] Modify bounds of the direction
- [x] Find translations in latent space
- [x] Saving pipeline for direction
- [x] Solve bald issue
- [x] Detect and improve bad translations
- [ ] Look for other repo to solve skin and age :construction:
- [ ] Solve specific issues (manually or with other methods)
- [x] Focused change by semantic segmentation
- [x] Resolve some artefacts and background problems with depth estimation
- [ ] Improve resolution (super resolution ?) :construction:

## Quick Start

### Installation

First, create a new virtual environment and install all the required packages:

```bash
pip install -e .
pip install -r requirements.txt
```

Then download the dataset of the competition available here: [drive dataset](https://drive.google.com/drive/folders/1-R1863MV8CuCjmycsLy05Uc6bdkWfuOP?usp=sharing)

And unzip the content in the `data/input_images` folder.

Then you can run the script `editor_API.py` to start the image editing API. Don't forget to change the config at the beginning of the script if you want to use the flexible config.

```bash
python editor_API.py
```

Or if you are using a Nvidia GPU:

```bash
FORCE_NATIVE=1 python editor_API.py
```

### Translations in latent space

To save the latent direction you want in the `data/` folder, you can click on the 'Save' button. **The name of the translation should follow the name convention:**

- for 'cursor' transformation (min - max): `{carac}_{'min' or 'max'}`. Example `Be_min` means cursor transformation to minimal bags under the eyes (Be_min).

- for default transformation (transformation that doesn't depend on the current value of the caracteristic): `{carac}_{new_value}`. Example: `Hc_0` means default transformation to black hair color (Hc_0)

- for specific transformation (transformation that depends on the current value of the caracteristic) that will overwrite default transformation: `{carac}_{new_value}_fr_{old_value}`. Example `A_0_fr_1` means specific transformation from intermediate age (A_1) to young age (A_0), 'fr' means 'from'.

**All required transformations must be treated for all input caracterisics to make a valid submission.**

Now, given an input image, the transformations to create caracterisitcs that are not in the initial image can automatically be got with the script `utils/translation.py`. Finally you can run the script `tools/translate.py` to generate the new images in a folder respecting the naming convention of the competition (the images are generated in `data/{latent_dir}/edited_images/`)

```bash
python tools/translate.py
```

Or if you are using a Nvidia GPU:

```bash
FORCE_NATIVE=1 tools/translate.py
```

**Of course, you can generate only a subset of images or caracteristics by removing some latent vectors or by removing keys of the dictionary containing the translations.**

If all the 72 images with all caracteristics are generated (1731 images in total), you can zip the folder containing the images and run the following script to check the submission (if all the required images are inside, with the good name, type, size...):

### Domain Mixup

To keep only the modification near the area where we expect them (called 'domain') for a specific change, a mixup is applied. The idea is to draw an area in a 512x512 white image that correspond to the area we expect add the changes, for each caracterisitc. The images are saved in `preprocess/mixup/domains/images/` (black and white images and black pixels correspond to the domain).

Then you need to compute the distance to the domain pixel per pixel. Don't forget to process only the domains you want by editing the script. It can take a while for each domain so copying the domains and the distances for similar caracteristics is highly recommended.

```bash
python tools/preprocess/mixup_compute_dist.py
```

Finally you can process the domain mixup. You can edit the paths in the script:

```bash
python tools/preprocess/mixup.py
```

By default, the resulting images are in `preprocess/mixup/edited_images_postmixum/`

### Segmentation

To go further in the last idea, we apply a semantic segmentation on the original image and the edited images in order to find for each image and for each transformation the area where we expect to find the change. First you need to get the segmentaion model by Git LFS (see [Git LFS](https://git-lfs.github.com/) for details):

```bash
git lfs pull
```

Then you can run the following script to merge the previously edited images with the original ones by semantic segmentation:

```bash
python tools/preprocess/segment.py
```
By default, the resulting images are in `preprocess/segmentation/edited_images_postsegmentation/`

### Depth estimation

To go further in this idea, we perform depth estimation to correct artefacts of the backgrounds for the images coming from the segmentation. First, you need to [download](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view) the depth estimation model and put it on `preprocess/depth_estimation/cp`.
The idea is to make a depth estimation with a transformer model (see [DPT](https://github.com/isl-org/DPT)). Then we build the foreground mask with a K-means algorithm. This allows to extract a relevant foreground from the segmented image and to paste it on the background of the original image. 

Then you can run the following script to merge the previously edited images with the original ones by depth estimation:

```bash
python tools/preprocess/estimate_depth.py
```
By default, the resulting images are in `preprocess/depth_estimation/edited_images_postdepth/`

### Check the submission

Checking the submission (number of files, names and format) can be easily checked with the following script:

```bash
python tools/check_submission.py <path_to_submission>
```

## AnyCost GAN tutorial

The tutorial of AnyCostGAN is in the file `AnycostGAN tuto.md` and the tutorial notebooks are in the `notebooks` folder. The original repository is here: [AnyCosGAN](https://github.com/mit-han-lab/anycost-gan). Note that the script `demo.py` was renamed to `editor_API.py` in this repository.

## Commit message

Commit messages are written in present tense (e.g. `:art: refactor training loop` instead of `:art: refactored training loop`).
They also start with one or two applicable emoji. This does not only look great but also makes you rethink what to add to a commit (one kind of action per commit!).

Make many but small commits!

| Emoji                                                     | Description                                      |
| --------------------------------------------------------- | ------------------------------------------------ |
| :tada: `:tada:`                                           | Initial Commit                                   |
| :sparkles: `:sparkles:`                                   | Add features                                     |
| :fire: `:fire:`                                           | Remove code or feature                           |
| :heavy_plus_sign: `:heavy_plus_sign:`                     | Add file (without adding features)               |
| :heavy_minus_sign: `:heavy_minus_sign:`                   | Remove file (without removing features)          |
| :beetle: `:beetle:`                                       | Fix bug                                          |
| :art: `:art:`                                             | Improve structure/format of code (including PEP) |
| :memo: `:memo:`                                           | Add/update docstring, comment or readme          |
| :rocket: `:rocket:`                                       | Improve performance                              |
| :pencil2: `:pencil2:`                                     | Fix typo                                         |
| :white_check_mark: `:white_check_mark:`                   | Add, update or pass tests                        |
| :arrow_up: `:arrow_up:`                                   | Update dependency or requirements                |
| :wrench: `:wrench:`                                       | Add/update configuration or configuration files  |
| :truck: `:truck:`                                         | Deplace or rename files or folders               |
| :construction: `:construction:`                           | Work in progress                                 |
| :twisted_rightwards_arrows: `:twisted_rightwards_arrows:` | Branch Merging                                   |
| :rewind: `:rewind:`                                       | Revert commit or changes                         |
| :speech_balloon: `:speech_balloon:`                       | Unknown category                                 |
