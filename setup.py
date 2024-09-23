from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pnnplus',
    version='0.1.1',
    description='A Python library that implements Parametric Neural Networks (PNN) for use in high-energy physics and beyond.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Xulei Sun',
    author_email='sxl66@outlook.com',
    url='https://github.com/feixukeji/pnnplus',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'tensorflow',
        'scikit-learn',
        'joblib',
        'matplotlib',
        'IPython'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='GPL-3.0',
)
