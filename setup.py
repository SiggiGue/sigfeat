# -- encoding: utf-8 --
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(name='sigfeat',
      version='0.2alpha',
      description='Signal Feature Extraction Framework.',
      author='Siegfried Gündert',
      author_email='siegfried.guendert@googlemail.com',
      license='BSD-3-Clause',
      packages=find_packages(exclude=('docs', '.git', '__pycache__')),
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      )
