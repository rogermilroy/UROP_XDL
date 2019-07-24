from setuptools import setup

setup(
    name='database',
    packages=['database',
              'database.database_manager',
              'database.base_model',
              'database.training_data',
              'database.training_run'],
    include_package_data=True,
    install_requires=[
    ],
)
