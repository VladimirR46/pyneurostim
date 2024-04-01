from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(name='pyneurostim',
      version='0.1',
      description='Analyzing neurophysiological data from NeuroStim',
      long_description=read_md('README.md'),
      url='https://github.com/VladimirR46/pyneurostim.git',
      author='Vladimir Antipov',
      author_email='vantipovm@gmail.com',
      packages=['pyneurostim'],
      package_dir={'pyneurostim': 'pyneurostim'},
      install_requires=[
          'mne','pyxdf','PySide6'
      ],
      zip_safe=False)