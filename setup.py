# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from setuptools import setup
from setuptools import find_packages

setup(name='dapeng',
      version='1.0.1',
      description='大棚 UNet-5',
      author='zhoujia',
      author_email='korewayume@foxmail.com',
      url='https://github.com/korewayume/dapeng',
      install_requires=['keras>=2.0.8', 'tensorflow>=1.4.0'],
      packages=find_packages())
