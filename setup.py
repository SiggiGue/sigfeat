# -- encoding: utf-8 --

from setuptools import setup, find_packages

setup(name='sigfeat',
      version='0.2alpha',
      description='Signal Feature Extraction Framework.',
      author='Siegfried Guendert',
      author_email='siegfried.guendert@googlemail.com',
      license='',
      packages=find_packages(exclude=('docs', '.git', '__pycache__')),
      )
