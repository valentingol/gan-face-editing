# Code adapted from https://github.com/mit-han-lab/anycost-gan
from functools import partial
import os
import sys
import time

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch

import anycostgan.models as models
from anycostgan.models.dynamic_channel import (set_uniform_channel_ratio,
                                               reset_generator)
from pipeline.utils.global_config import GlobalConfig


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        t1 = time.time()
        ret = self.fn(*self.args, **self.kwargs)
        t2 = time.time()
        self.signals.result.emit((ret, t2 - t1))


class FaceEditor(QMainWindow):
    def __init__(self, anycost_config, default_max_value, custom_max_value):
        super().__init__()
        self.anycost_config = anycost_config
        # Load assets
        self.load_assets(default_max_value, custom_max_value)
        # Title
        self.setWindowTitle('Face Editing with Anycost GAN')
        # Window size
        self.setFixedSize(1800, 1200)

        # Plot the original image
        self.original_image = QLabel(self)
        self.set_img_location(self.original_image, 100, 72, 360, 360)
        pixmap = self.np2pixmap(self.org_image_list[0])
        self.original_image.setPixmap(pixmap)
        self.original_image_label = QLabel(self)
        self.original_image_label.setText('original')
        self.set_text_format(self.original_image_label)
        self.original_image_label.move(230, 42)

        # Display the edited image
        self.edited_image = QLabel(self)
        self.set_img_location(self.edited_image, 700, 72, 360, 360)
        self.projected_image = self.generate_image()
        self.edited_image.setPixmap(self.projected_image)
        self.edited_image_label = QLabel(self)
        self.edited_image_label.setText('projected')
        self.set_text_format(self.edited_image_label)
        self.edited_image_label.move(830, 42)

        # Build the sample list
        drop_list = QComboBox(self)
        drop_list.addItems(self.file_names)
        drop_list.currentIndexChanged.connect(self.select_image)
        drop_list.setGeometry(100, 490, 200, 30)
        drop_list.setCurrentIndex(0)
        drop_list_label = QLabel(self)
        drop_list_label.setText('* select sample:')
        self.set_text_format(drop_list_label, 'left', 15)
        drop_list_label.setGeometry(100, 470, 200, 30)

        # Build editing sliders
        self.attr_sliders = dict()
        for i_slider, key in enumerate(self.direction_dict.keys()):
            if i_slider < 18:
                tick_label = QLabel(self)
                tick_label.setText('|')
                self.set_text_format(tick_label, 'center', 10)
                tick_label.setGeometry(700 + 175, 470 + i_slider * 30 + 9,
                                       50, 20)

                this_slider = QSlider(Qt.Horizontal, self)
                this_slider.setGeometry(700, 470 + i_slider * 30, 400, 30)
                this_slider.sliderReleased.connect(self.slider_update)
                this_slider.setMinimum(-100)
                this_slider.setMaximum(100)
                this_slider.setValue(0)
                self.attr_sliders[key] = this_slider

                attr_label = QLabel(self)
                attr_label.setText(key)
                self.set_text_format(attr_label, 'right', 13)
                attr_label.move(700 - 110, 470 + i_slider * 30 + 2)
            else:
                tick_label = QLabel(self)
                tick_label.setText('|')
                self.set_text_format(tick_label, 'center', 10)
                tick_label.setGeometry(1300 + 175,
                                       470 + (i_slider - 18) * 30 + 9, 50, 20)

                this_slider = QSlider(Qt.Horizontal, self)
                this_slider.setGeometry(1300, 470 + (i_slider - 18) * 30,
                                        400, 30)
                this_slider.sliderReleased.connect(self.slider_update)
                this_slider.setMinimum(-100)
                this_slider.setMaximum(100)
                this_slider.setValue(0)
                self.attr_sliders[key] = this_slider

                attr_label = QLabel(self)
                attr_label.setText(key)
                self.set_text_format(attr_label, 'right', 13)
                attr_label.move(1300 - 110, 470 + (i_slider - 18) * 30 + 2)

        # Build models sliders
        base_h = 560
        channel_label = QLabel(self)
        channel_label.setText('channel:')
        self.set_text_format(channel_label, 'left', 13)
        channel_label.setGeometry(100, base_h + 5, 100, 30)

        self.channel_slider = QSlider(Qt.Horizontal, self)
        self.channel_slider.setGeometry(190, base_h, 210, 30)
        self.channel_slider.sliderReleased.connect(self.model_update)
        self.channel_slider.setMinimum(0)
        self.channel_slider.setMaximum(3)
        self.channel_slider.setValue(3)
        for i, text in enumerate(['1/4', '1/2', '3/4', '1']):
            channel_label = QLabel(self)
            channel_label.setText(text)
            self.set_text_format(channel_label, 'center', 13)
            channel_label.setGeometry(190 + i * 63 - 50 // 2 + 10, base_h + 20,
                                      50, 20)

        resolution_label = QLabel(self)
        resolution_label.setText('resolution:')
        self.set_text_format(resolution_label, 'left', 13)
        resolution_label.setGeometry(100, base_h + 55, 100, 30)

        self.resolution_slider = QSlider(Qt.Horizontal, self)
        self.resolution_slider.setGeometry(190, base_h + 50, 210, 30)
        self.resolution_slider.sliderReleased.connect(self.model_update)
        self.resolution_slider.setMinimum(0)
        self.resolution_slider.setMaximum(3)
        self.resolution_slider.setValue(3)
        for i, text in enumerate(['128', '256', '512', '1024']):
            resolution_label = QLabel(self)
            resolution_label.setText(text)
            self.set_text_format(resolution_label, 'center', 10)
            resolution_label.setGeometry(190 + i * 63 - 50 // 2 + 10,
                                         base_h + 70, 50, 20)

        # Build button slider
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.move(100, 700)
        self.reset_button.clicked.connect(self.reset_clicked)

        # Build button slider
        self.save_button = QPushButton('Save trans', self)
        self.save_button.move(280, 700)
        self.save_button.clicked.connect(partial(self.slider_update,
                                                 force_full_g=True,
                                                 save_trans=True,
                                                 save_img=False))

        # Button for saving image
        self.save_img_button = QPushButton('Save img', self)
        self.save_img_button.move(280, 760)
        self.save_img_button.clicked.connect(partial(self.slider_update,
                                                     force_full_g=True,
                                                     save_trans=False,
                                                     save_img=True))

        # Add loading gif
        # Create label
        self.loading_label = QLabel(self)
        self.loading_label.setGeometry(500 - 25, 240, 50, 50)

        self.loading_label.setObjectName("label")
        self.movie = QMovie("ressources/loading.gif")
        self.loading_label.setMovie(self.movie)
        self.movie.start()
        self.movie.setScaledSize(QSize(50, 50))
        self.loading_label.setVisible(False)

        # Extra time stat
        self.time_label = QLabel(self)
        self.time_label.setText('')
        self.set_text_format(self.time_label, 'center', 18)
        self.time_label.setGeometry(500 - 25, 240, 50, 50)

        # Status bar
        self.statusBar().showMessage('Ready.')

        # Multi-thread
        self.thread_pool = QThreadPool()

        self.show()

    def load_assets(self, default_max_value, custom_max_values):
        self.anycost_channel = 1.0
        self.anycost_resolution = 1024

        # Build the generator
        self.generator = models.get_pretrained('generator',
                                               self.anycost_config).to('cpu')
        self.generator.eval()
        self.mean_latent = self.generator.mean_style(10000)

        # Select only a subset of the directions to use
        '''
        possible keys:
        ['00_5_o_Clock_Shadow', '01_Arched_Eyebrows', '02_Attractive',
         '03_Bags_Under_Eyes', '04_Bald', '05_Bangs', '06_Big_Lips',
         '07_Big_Nose', '08_Black_Hair', '09_Blond_Hair', '10_Blurry',
         '11_Brown_Hair', '12_Bushy_Eyebrows', '13_Chubby', '14_Double_Chin',
         '15_Eyeglasses', '16_Goatee', '17_Gray_Hair', '18_Heavy_Makeup',
         '19_High_Cheekbones', '20_Male', '21_Mouth_Slightly_Open',
         '22_Mustache', '23_Narrow_Eyes', '24_No_Beard', '25_Oval_Face',
         '26_Pale_Skin', '27_Pointy_Nose', '28_Receding_Hairline',
         '29_Rosy_Cheeks', '30_Sideburns', '31_Smiling', '32_Straight_Hair',
         '33_Wavy_Hair', '34_Wearing_Earrings', '35_Wearing_Hat',
         '36_Wearing_Lipstick', '37_Wearing_Necklace',
         '38_Wearing_Necktie', '39_Young']
        '''

        direction_map = {
            'skin': '26_Pale_Skin',
            'age': '39_Young',
            'sexe': '20_Male',
            'bangs': '05_Bangs',
            'pointy nose': '27_Pointy_Nose',
            'black hair': '08_Black_Hair',
            'blond hair': '09_Blond_Hair',
            'brown hair': '11_Brown_Hair',
            'gray hair': '17_Gray_Hair',
            'bald': '04_Bald',
            'double chin': '14_Double_Chin',
            'straight hair': '32_Straight_Hair',
            'curly hair': '33_Wavy_Hair',
            'eyes bags': '03_Bags_Under_Eyes',
            'narrow eyes': '23_Narrow_Eyes',
            'pointy nose': '27_Pointy_Nose',
            'lips size': '06_Big_Lips',
            'nose_size': '07_Big_Nose',
            'chubby': '13_Chubby',
            'attractive': '02_Attractive',
            'blurry': '10_Blurry',
            'eyebrows': '12_Bushy_Eyebrows',
            'eyeglasses': '15_Eyeglasses',
            'goatee': '16_Goatee',
            'makup': '18_Heavy_Makeup',
            'mouth open': '21_Mouth_Slightly_Open',
            'mustache': '22_Mustache',
            'no beard': '24_No_Beard',
            'oval face': '25_Oval_Face',
            'hairline': '28_Receding_Hairline',
            'sideburns': '30_Sideburns',
            'smile': '31_Smiling',
            'lipstick': '36_Wearing_Lipstick',
            'rosy cheeks': '29_Rosy_Cheeks',
        }
        max_values = {k: default_max_value for k in direction_map.keys()}
        max_values = {**max_values, **custom_max_values}
        self.max_values = max_values

        boundaries = models.get_pretrained('boundary', self.anycost_config)
        self.direction_dict = dict()
        for k, v in direction_map.items():
            self.direction_dict[k] = boundaries[v].view(1, 1, -1)

        # 3. Prepare the latent code and original images
        file_names = sorted(os.listdir(DATA_DIR))
        self.file_names = [f for f in file_names
                           if f.endswith('.png') or f.endswith('.jpg')]
        self.latent_code_list = []
        self.org_image_list = []

        for fname in self.file_names:
            org_image = np.asarray(Image.open(
                os.path.join(DATA_DIR, fname)
                ).convert('RGB'))
            npy_name = fname.replace('.jpg', '.npy').replace('.png', '.npy')
            latent_code = torch.from_numpy(
                np.load(os.path.join(PROJECTION_DIR, 'projected_latents',
                                     npy_name)))
            self.org_image_list.append(org_image)
            self.latent_code_list.append(latent_code.view(1, -1, 512))

        # Set up the initial display
        self.sample_idx = 0
        self.org_latent_code = self.latent_code_list[self.sample_idx]

        # Input kwargs for the generator
        self.input_kwargs = {'styles': self.org_latent_code, 'noise': None,
                             'randomize_noise': False,
                             'input_is_style': True}

    @staticmethod
    def np2pixmap(np_arr):
        height, width, _ = np_arr.shape
        q_image = QImage(np_arr.data, width, height, 3 * width,
                         QImage.Format_RGB888)
        return QPixmap(q_image)

    @staticmethod
    def set_img_location(img_op, x, y, w, h):
        img_op.setScaledContents(True)
        img_op.setFixedSize(w, h)  # w, h
        img_op.move(x, y)  # x, y

    @staticmethod
    def set_text_format(text_op, align='center', font_size=18):
        if align == 'center':
            align = Qt.AlignCenter
        elif align == 'left':
            align = Qt.AlignLeft
        elif align == 'right':
            align = Qt.AlignRight
        else:
            raise NotImplementedError
        text_op.setAlignment(align)
        text_op.setFont(QFont('Arial', font_size))

    def select_image(self, idx):
        self.sample_idx = idx
        self.org_latent_code = self.latent_code_list[self.sample_idx]
        pixmap = self.np2pixmap(self.org_image_list[self.sample_idx])
        self.original_image.setPixmap(pixmap)
        self.input_kwargs['styles'] = self.org_latent_code
        self.projected_image = self.generate_image()
        self.edited_image.setPixmap(self.projected_image)
        self.reset_sliders()

    def reset_sliders(self):
        for slider in self.attr_sliders.values():
            slider.setValue(0)
        self.edited_image_label.setText('projected')
        self.statusBar().showMessage('Ready.')
        self.time_label.setText('')

    def generate_image(self, pixmap=True):
        def image_to_np(x):
            assert x.shape[0] == 1
            x = x.squeeze(0).permute(1, 2, 0)
            x = (x + 1) * 0.5  # 0-1
            x = (x * 255).cpu().numpy().astype('uint8')
            return x

        with torch.no_grad():
            out = self.generator(**self.input_kwargs)[0].clamp(-1, 1)
            out = image_to_np(out)
            out = np.ascontiguousarray(out)
            if pixmap:
                return self.np2pixmap(out)
            else:
                return out

    def set_sliders_status(self, active):
        for slider in self.attr_sliders.values():
            slider.setEnabled(active)

    def slider_update(self, force_full_g=True, save_trans=False,
                      save_img=False):
        self.set_sliders_status(False)
        self.statusBar().showMessage('Running...')
        self.time_label.setText('')
        self.loading_label.setVisible(True)
        edited_code = self.org_latent_code.clone()
        for direction_name in self.attr_sliders.keys():
            edited_code[:, : N_STYLE_TO_CHANGE] = \
                edited_code[:, : N_STYLE_TO_CHANGE] \
                + self.attr_sliders[direction_name].value() \
                * self.direction_dict[direction_name] / 100 \
                * self.max_values[direction_name]
        self.input_kwargs['styles'] = edited_code

        if save_trans:
            translation = edited_code - self.org_latent_code
            text, _ = QInputDialog.getText(self, "Name of translation",
                                           "Name:",
                                           QLineEdit.Normal, "")
            path = os.path.join(PROJECTION_DIR, 'translations_vect',
                                text + '.npy')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            np.save(path, translation.cpu().numpy())
            print(f'Direction "{text}" saved at {path}')

        if save_img:
            image = self.generate_image(pixmap=False)
            image = Image.fromarray(image)
            image = image.resize((512, 512), Image.ANTIALIAS)
            text, _ = QInputDialog.getText(self, "Name of image", "Name:",
                                           QLineEdit.Normal, "")
            base_name = self.file_names[self.sample_idx].split('.')[0]
            dir_path = os.path.join(PROJECTION_DIR, 'manual_edited_images',
                                    base_name)
            path = os.path.join(dir_path, text + '.png')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            image.save(path)
            print('Image saved at', path)

        if not force_full_g:
            set_uniform_channel_ratio(self.generator, self.anycost_channel)
            self.generator.target_res = self.anycost_resolution

        # Generate the images in a separate thread
        worker = Worker(partial(self.generate_image, pixmap=True))
        worker.signals.result.connect(self.after_slider_update)
        self.thread_pool.start(worker)

    def after_slider_update(self, ret):
        edited, used_time = ret
        self.edited_image.setPixmap(edited)

        reset_generator(self.generator)
        self.edited_image_label.setText('edited')
        self.statusBar().showMessage('Done in {:.2f}s'.format(used_time))
        self.time_label.setText('{:.2f}s'.format(used_time))
        self.set_sliders_status(True)
        self.loading_label.setVisible(False)

    def model_update(self):
        channel_slider_num = self.channel_slider.value()
        resolution_slider_num = self.resolution_slider.value()
        self.anycost_channel = [0.25, 0.5, 0.75, 1.0][channel_slider_num]
        self.anycost_resolution = [128, 256, 512, 1024][resolution_slider_num]

    def reset_clicked(self):
        self.reset_sliders()
        self.edited_image.setPixmap(self.projected_image)


if __name__ == '__main__':
    config = GlobalConfig.build_from_argv(fallback='configs/exp/base.yaml')

    # Get the configs
    DATA_DIR = config.data_dir
    PROJECTION_DIR = config.projection_dir
    N_STYLE_TO_CHANGE = config.editor.n_style_to_change
    default_max_val = config.editor.default_max_value
    custom_max_val = config.editor.custom_max_value

    flexible_config = config.anycost_config_flexible
    anycost_config = ('anycost-ffhq-config-f-flexible'
                      if flexible_config else 'anycost-ffhq-config-f')

    # Run the editor
    app = QApplication(sys.argv[0:1])
    ex = FaceEditor(anycost_config, default_max_val, custom_max_val)
    sys.exit(app.exec_())
