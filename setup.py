from os import path

from setuptools import setup, find_packages

# find version
with open('ffsas/__init__.py') as f:
    version = f.read().splitlines()[-1].split("'")[-2]

# framework dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# read contents from README.md
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# setup
setup(
    name='ffsas',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license='BSD-3-Clause License',
    author='Kuangdai Leng, Steve King, Tim Snow, Sarah Rogers, '
           'Jeyan Thiyagalingam',
    author_email='kuangdai.leng@stfc.ac.uk',
    description='Free-form inversion for small-angle scattering',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
