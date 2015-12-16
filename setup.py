# http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/#uploading-your-project-to-pypi
# to publish package:
# 1) python setup.py register
# 2) python setup.py sdist bdist_wheel upload

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

def readme():
    with open('README.md') as f:
        return f.read()

def install_libfm():
    import subprocess
    subprocess.call('rm -rf pywFM/libfm', shell=True)
    subprocess.call('git clone https://github.com/srendle/libfm pywFM/libfm', shell=True)
    subprocess.call('cd pywFM/libfm && make', shell=True)

class PywfmInstall(install):
    def run(self):
        install_libfm()
        install.run(self)

class PywfmDevelop(develop):
    def run(self):
        install_libfm()
        develop.run(self)

setup(name='pywFM',
      version='0.5',
      description='Python wrapper for libFM',
      long_description='Python wrapper for the official libFM (http://libfm.org/)',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords='python wrapper libfm factorization machines',
      url='http://github.com/jfloff/pywFM',
      author='Joao Loff',
      author_email='jfloff@gmail.com',
      license='MIT',
      packages=['pywFM'],
      install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas'
      ],
      zip_safe=False,
      cmdclass={
        'install': PywfmInstall,
        'develop': PywfmDevelop
      },
      package_data={
        'pywFM': ['libfm/bin/libFM']
      })
