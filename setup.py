from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = 'SGTL - Spectral Graph Theory Library'
LONG_DESCRIPTION =\
    "This library provides several methods and algorithms relating to spectral graph theory in python."

# Setting up
setup(
    name="sgtl",
    version=VERSION,
    author="Peter Macgregor",
    author_email="<macgregor.pr@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy", "scipy", "sklearn"],

    keywords=['python', 'spectral', 'graph', 'algorithms', 'clustering', 'cheeger'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        'Operating System :: POSIX :: Linux'
    ]
)
