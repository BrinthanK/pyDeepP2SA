from setuptools import setup

classifiers = [
  'Development Status :: 4 - Beta',
  'Intended Audience :: Education',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3',
  'Topic :: Scientific/Engineering'
]

install_requires = [
    'numpy',
    'scikit-image',
    'torch',
    'torchvision',
    'opencv-python',
    'matplotlib',
    'pandas',
    'seaborn'
]

setup(
    name='pyDeepP2SA',
    version='0.0.7',
    description='A python package for particle size and shape analysis using deep learning',
    py_modules=["pyDeepP2SA"],
    package_dir={'': 'src'},
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BrinthanK/pyDeepP2SA',
    author='Brinthan K',
    author_email='kanesalingambrinthan187@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Particle shape analyser, PSD, Deep learning, Image processing, Circularity',
    install_requires=install_requires
    )
