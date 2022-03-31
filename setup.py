import os
from setuptools import setup, find_packages

# Installation
config = {
    'name': 'InterCentrales',
    'version': '1.0',
    'description': 'Inter Centrales Ceteris Paribus face challenge.',
    'author': 'Valentin Goldite',
    'author_email': 'valentin.goldite@gmail.com',
    'packages': find_packages(),
    'zip_safe': True
}

setup(**config)
