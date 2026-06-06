from setuptools import setup, find_packages

setup(name='gym-env',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym', 'airsim']
)
