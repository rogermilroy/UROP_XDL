from setuptools import setup, find_packages

setup(
    name='testing',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.14.5',
        'torch>=1.1.0',
        'torchvision>=0.2.1',
        'pyzmq>=18.0.2',
        'tqdm>=4.31.1',
        'pandas>=0.23.4'
    ],
)
