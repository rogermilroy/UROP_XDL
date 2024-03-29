from setuptools import setup, find_packages

setup(
    name='utils',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.14.5',
        'torch>=1.5.0',
        'torchvision>=0.2.1',
        'pyzmq>=18.0.2'
    ],
)
