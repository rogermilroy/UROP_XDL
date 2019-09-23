from setuptools import setup

setup(
    name='visualisation',
    packages=['server'],
    include_package_data=True,
    install_requires=[
        'pyzmq>=18.0.2',
        'pymongo>=3.8.0',
        'ujson>=1.35',
        'tqdm>=4.31.1',
        'datalake>=0.0.0',
        'Flask>=1.0.2',
        'Flask-RESTful>=0.3.6'
    ],
)
