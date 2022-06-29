"""Setup of gan-face-editing."""

from setuptools import find_packages, setup

# Installation
config = {
    'name': 'gan-face-editing',
    'version': '1.1.1',
    'description': 'Inter Centrales Ceteris Paribus face challenge.',
    'author': 'Valentin Goldite',
    'author_email': 'valentin.goldite@gmail.com',
    'packages': find_packages(),
    'zip_safe': True
}

setup(**config)
