from setuptools import setup, find_packages

setup(
    name='phytorch-lib',
    version='0.1.1',
    author='Tong Lei, Kyle T. Rizzo, Brian N. Bailey',
    author_email='',
    description='PhyTorch is a PyTorch-based modeling toolkit for fitting common plant physiological models of photosynthesis, stomatal conductance, leaf hydraulics, and optical properties.',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/ktrizzo/phytorch',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'phytorch': [
            'data/tests/*.csv',
            'data/fvcb/**/*.csv',
            'data/stomatal/**/*.csv',
            'leafoptics/*.txt',
        ],
    },
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'scipy>=1.5.0',
        'matplotlib>=3.3.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    keywords='photosynthesis stomatal-conductance hydraulics optics plant-physiology pytorch',
)
