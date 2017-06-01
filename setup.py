#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='pysembler',
    version='0.1.0',
    description="An automatic ensembler of machine learning models in python",
    author="Abhishek Thakur",
    author_email='abhishek4@gmail.com',
    url='https://github.com/abhishekkrthakur/pysembler',
    packages=[
        'pysembler',
    ],
    package_dir={'pysembler':
                 'pysembler'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='pysembler',
    classifiers=[
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
