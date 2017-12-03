# http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/#uploading-your-project-to-pypi
# to publish package:
# 1) python setup.py register
# 2) python setup.py sdist bdist_wheel upload
# 3) Convert pypi documentation (http://devotter.com/converter)

from setuptools import setup

setup(name='pywFM',
      version='0.12.3',
      description='Python wrapper for libFM',
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
      zip_safe=False)
