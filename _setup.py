from setuptools import setup, find_packages
import os

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
        return f.read()

setup(
    name = 'gait_modulation',
    version = '1.0',
    description = '<DESCRIPTION FOR GAIT MODULATION>',
    long_description=read_file('README.rst'),
    author='Mohammad Orabe',
    author_email='orabe.mhd@gmail.com',
    packages = find_packages(),
    # install_requires=read_file('requirements.txt').splitlines(),
)