"""
setup script.
"""

from setuptools import setup, find_packages

setup(
    name='dubstandard',
    version='0.1.0',
    description='Calculates optimum Dubinin pore volume from an isotherm',
    url='https://github.com/sblanky/dubstandard',
    install_requires=[],
    author='L. Scott Blankenship',
    author_email='leo.blankenship1@nottingham.ac.uk',
    packages=find_packages(),
    zip_safe=False,
    entry_points={

    },
)
