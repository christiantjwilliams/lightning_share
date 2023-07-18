"""
This file contains setup requirements for the drvr python package.
"""

from setuptools import setup

setup(name='hardware_control',
      version='0.3',
      description='Python software based on pyVISA library for controlling various lab equipemnt through GPIB.',
      url='https://github.mit.edu/n-h/hardware_control',
      author='QPLAB',
      author_email='harris.nicholasc@gmail.com',
      license='MIT',
      packages=['hardware_control'],
      install_requires=[
      ],
      zip_safe=False)
