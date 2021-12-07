"""Setup script for realpython-reader"""

import os.path
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

# This call to setup() does all the work
setup(
    name="GN0",
    version="1.0.0",
    description="Graph Network based board game engine",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yannikkellerde/Gabor_Graph_Networks",
    author="Yannik Keller",
    author_email="yannik@kelnet.de",
    license="GPL-3",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["graph_game","GN0"],
    include_package_data=True,
    entry_points={},
)