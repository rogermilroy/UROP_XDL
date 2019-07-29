from setuptools import setup

setup(
    name='datalake',
    packages=['datalake'],
    include_package_data=True,
    install_requires=[
        'numpy>=1.14.5',
        'torch>=1.1.0',
        'torchvision>=0.2.1',
        'pyzmq>=18.0.2',
        'pymongo>=3.8.0'
    ],
)
