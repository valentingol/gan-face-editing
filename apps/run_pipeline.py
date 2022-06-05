import os

if __name__ == '__main__':
    # Translate images in latent space
    os.system('FORCE_NATIVE=1 python3 pipeline/translate.py')
    # Apply domain mixup
    os.system('python3 pipeline/domain_mixup.py')
    # Apply segmentation
    os.system('python3 pipeline/segment.py')
    # Apply depth estimation
    os.system('python3 pipeline/depth_segmentation.py')
