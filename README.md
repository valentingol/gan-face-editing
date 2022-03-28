# InterCentrales Competition 2022

Repository of Inter-Centrales 2022 AI competition: Ceteris Paribus Face Challenge: [site of the competition](https://transfer-learning.org/competition.html).

![alt text](assets/figures/compet_img.png)

## To-Do list

- [x] <span style="color:green"> Project images in latent space <span>
- [x] <span style="color:green"> Modify bounds of the direction <span style="color:green">
- [ ] Find translations in latent space
- [x] <span style="color:green"> Saving pipeline image + direction <span style="color:green">
- [x] <span style="color:green"> Solve bald issue <span style="color:green">
- [ ] Solve specific issues
- [ ] Fix artifacts (using original images)
- [ ] Improve resolution (super resolution ?)

## To start

First create a new virtual environment and install all the required packages:

```bash
pip install -r requirements.txt
```

Then download the dataset of the competition available here: [drive dataset](https://drive.google.com/drive/folders/1-R1863MV8CuCjmycsLy05Uc6bdkWfuOP?usp=sharing)

And unzip the content in the `data/input_images` folder.

Then you can run the script `demo.py` to start the image editing API. Don't forget to change the config at the beginning of the script if you want to use the flexible config.

```bash
python demo.py
```

Or if you are using a Nvidia GPU:

```bash
FORCE_NATIVE=1 python demo.py
```

To save the latent direction you want in the `data/` folder, you can click on the 'Save' button. It can be used to automatically generate edited images using this translation via the scripts `tools/translate.py` and `utils/translation.py` (work in progress :construction:).

The tutorial of AnyCostGAN is in the file `AnycostGAN tuto.md` and the tutorial notebooks are in the `notebooks` folder. The original repository is here: [AnyCosGAN](https://github.com/mit-han-lab/anycost-gan).

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
| :zap: `:zap:`                                             | Improve performance                              |
| :pencil2: `:pencil2:`                                     | Fix typo                                         |
| :white_check_mark: `:white_check_mark:`                   | Add, update or pass tests                        |
| :arrow_up: `:arrow_up:`                                   | Update dependency or requirements                |
| :wrench: `:wrench:`                                       | Add/update configuration or configuration files  |
| :truck: `:truck:`                                         | Deplace or rename files or folders               |
| :construction: `:construction:`                           | Work in progress                                 |
| :twisted_rightwards_arrows: `:twisted_rightwards_arrows:` | Branch Merging                                   |
| :rewind: `:rewind:`                                       | Revert commit or changes                         |
| :speech_balloon: `:speech_balloon:`                       | Unknown category                                 |
