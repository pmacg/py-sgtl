# Contributing to the SGTL

There are 2 high level processes when developing this project:
* develop an enhancement, and merge it to the next release branch
* make a new release

This document provides a checklist for the maintainer of the library to follow.

The next release of the library is tracked by the `git` branch `next-release`.

## Developing an enhancement or bugfix
Please follow this checklist.
* Check out a new branch for the enhancement or bugfix from the branch `next-release`.
* Write tests for the new functionality you are developing or the bug you are fixing.
* Develop the code for the new functionality or bugfix.
* Make sure the new code is well documented - full docstrings and comments in the code.
* Add an entry to the `changelog.rst` file in the documentation folder.
* Run the `check.sh` script to run all of the python tests and to build the documentation, ensure this script succeeds.
* Check the generated documentation for the new development.
* Merge the development branch to the `next-release` branch.

## Releasing a new version of the library
Please follow this checklist
* Check out a new release branch from the branch `next-release`.
* Update the version number of the release in `conf.py` for the documentation and `setup.py` for PyPI.
* Update the `changelog.rst` file with the name and date of the release.
* Run the `check.sh` script to run all tests and check they pass.
* Check the documentation is built correctly - check changelog
* Push the release branch to github.
* Build the python package with `python3 setup.py sdist bdist_wheel`.
* Upload the package with `twine upload dist/*`.
* Merge the release branch to the `main` branch.
* Merge the main branch to the `next-release` branch.
