from setuptools import setup, find_packages

setup(
    name='visualisation',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pyzmq>=18.0.2',
        'pymongo>=3.8.0',
        'ujson>=3.2.0',
        'tqdm>=4.31.1',
        'datalake>=0.0.0',
        'Flask>=1.0.2',
        'Flask-RESTful>=0.3.6'
        'flask-cors>=3.0.8'
    ],
)
