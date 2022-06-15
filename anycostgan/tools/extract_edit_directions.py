# Code from https://github.com/mit-han-lab/anycost-gan
"""Get edit directions for FFHQ models."""

import horovod.torch as hvd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from anycostgan import models
from anycostgan.thirdparty.manipulator import project_boundary, train_boundary

# configurations for the job
DEVICE = 'cuda'
# specify the attributes to compute latent direction
chosen_attr = ['Smiling', 'Young', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
               'Eyeglasses', 'Mustache']
attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
             'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
             'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
             'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
             'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
             'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
             'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
             'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
             'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
             'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
             'Young']
SPACE = 'w'  # chosen from ['z', 'w', 'w+']
CONFIG = 'anycost-ffhq-config-f'


@torch.no_grad()
def get_style_attribute_pairs():
    """Get style and attribute pairs."""
    # NOTE: This function is written with horovod to accelerate
    # the extraction (by n_gpu times)
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(hvd.rank() * 999 + 1)
    if hvd.rank() == 0:
        print(' * Extracting style-attribute pairs...')
    # build and load the pre-trained attribute predictor on CelebA-HQ
    predictor = models.get_pretrained('attribute-predictor').to(DEVICE)
    # build and load the pre-trained anycost generator
    generator = models.get_pretrained('generator', CONFIG).to(DEVICE)

    predictor.eval()
    generator.eval()

    # randomly generate images and feed them to the predictor
    # configs from https://github.com/genforce/interfacegan
    randomized_noise = False
    truncation_psi = 0.7
    batch_size = 16
    n_batch = 500000 // (batch_size * hvd.size())

    styles = []
    attributes = []

    mean_style = generator.mean_style(100000).view(1, 1, -1)
    assert SPACE in ['w', 'w+', 'z']
    for _ in tqdm(range(n_batch), disable=hvd.rank() != 0):
        if SPACE in ['w', 'z']:
            z = torch.randn(batch_size, 1, generator.style_dim, device=DEVICE)
        else:
            z = torch.randn(batch_size, generator.n_style, generator.style_dim,
                            device=DEVICE)
        images, w = generator(z,
                              return_styles=True,
                              truncation=truncation_psi,
                              truncation_style=mean_style,
                              input_is_style=False,
                              randomize_noise=randomized_noise)
        images = F.interpolate(images.clamp(-1, 1), size=256, mode='bilinear',
                               align_corners=True)
        attr = predictor(images)
        # move to cpu to save memory
        if SPACE == 'w+':
            styles.append(w.to('cpu'))
        elif SPACE == 'w':
            # Originally duplicated
            styles.append(w.mean(1, keepdim=True).to('cpu'))
        else:
            styles.append(z.to('cpu'))
        attributes.append(attr.to('cpu'))

    styles = torch.cat(styles, dim=0)
    attributes = torch.cat(attributes, dim=0)

    styles = hvd.allgather(styles, name='styles')
    attributes = hvd.allgather(attributes, name='attributes')
    if hvd.rank() == 0:
        print(styles.shape, attributes.shape)
        torch.save(attributes, f'attributes_{CONFIG}.pt')
        torch.save(styles, f'styles_{CONFIG}.pt')


def extract_boundaries():
    """Extract boundaries for the style-attribute pairs."""
    styles = torch.load(f'styles_{CONFIG}.pt')
    attributes = torch.load(f'attributes_{CONFIG}.pt')
    attributes = attributes.view(-1, 40, 2)
    # Probability to be positive [n, 40]
    prob = F.softmax(attributes, dim=-1)[:, :, 1]

    boundaries = {}
    for idx, attr in tqdm(enumerate(attr_list), total=len(attr_list)):
        this_prob = prob[:, idx]

        boundary = train_boundary(latent_codes=styles.squeeze().cpu().numpy(),
                                  scores=this_prob.view(-1, 1).cpu().numpy(),
                                  chosen_num_or_ratio=0.02,
                                  split_ratio=0.7,
                                  )
        key_name = f'{idx:02d}' + '_' + attr
        boundaries[key_name] = boundary

    boundaries = {k: torch.tensor(v) for k, v in boundaries.items()}
    torch.save(boundaries, f'boundaries_{CONFIG}.pt')


# experimental; not yet used in the demo
# do not observe significant improvement right now
def project_boundaries():  # only project the ones used for demo
    """Project boundaries to the latent space and save them."""
    boundaries = torch.load(f'boundaries_{CONFIG}.pt')
    chosen_idx = [attr_list.index(attr) for attr in chosen_attr]
    sorted_keys = [f'{idx:02d}' + '_' + attr_list[idx]
                   for idx in chosen_idx]
    all_boundaries = np.concatenate([boundaries[k].cpu().numpy()
                                     for k in sorted_keys])  # n, 512
    similarity = all_boundaries @ all_boundaries.T
    projected_boundaries = []
    for i_b in range(len(sorted_keys)):
        # NOTE: the number of conditions is exponential;
        # we only take the 2 most related boundaries
        this_sim = similarity[i_b]
        this_sim[i_b] = -100.  # exclude self
        idx1, idx2 = np.argsort(this_sim)[-2:]  # most similar 2
        projected_boundaries.append(project_boundary(
            all_boundaries[i_b][None], all_boundaries[idx1][None],
            all_boundaries[idx2][None]))
    boundaries = dict(zip(sorted_keys, torch.tensor(projected_boundaries)))
    torch.save(boundaries, f'boundary_projected_{CONFIG}.pt')


if __name__ == '__main__':
    # get_style_attribute_pairs()
    extract_boundaries()
    # project_boundaries()
