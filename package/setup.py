from setuptools import setup

setup(
    name="mlflow_pkg",
    version="0.1",
    author="Thien Nhan",
    author_email="nhanthien.tnn@gmail.com",
    packages=['package.feature','package.ml_training'],
    install_requires=['numpy','pandas','scikit-learn','matplotlib','mlflow']
)