from setuptools import setup
from setuptools.command.install import install

def readme():
    with open('README.md') as f:
        return f.read()

class Installer(install):

    def run(self):
        import subprocess
        subprocess.call(['./install-libfm.sh'])
        install.run(self)

setup(name='pywFM',
      version='0.3',
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
        'sklearn'
      ],
      zip_safe=False,
      cmdclass={'install': Installer},
      package_data={
        'pywFM': ['libfm/bin/*']
      })
