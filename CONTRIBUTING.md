# Contributing to the SGTL

The next release of the library is tracked by the `git` branch `next-release`.

## Developing an enhancement or bugfix
Thanks for your interest in contributing to this project. Please follow the following process when contributing.

* Ensure there is an issue on the Github project for the development you would like to do. Feel free to create an issue describing the change you will make.
* Fork the project.
* Check out a new branch for the enhancement or bugfix **from the branch `next-release`**.
* Write tests for the new functionality you are developing or the bug you are fixing.
* Develop the code for the new functionality or bugfix.
* Make sure the new code is well documented - full docstrings and comments in the code.
* Add an entry to the `changelog.rst` file in the documentation folder.
  * If there is no section for the next release yet, add a new release section named 'Unreleased'.
* Run the `check.sh` script to run all of the python tests and to build the documentation, ensure this script succeeds.
* Check the generated documentation for the new development.
* Commit all of your changes, including the newly generated documentation files, if any.
* Submit a pull request from your development branch into the `next-releaes` branch of the main project.

## Releasing a new version of the library
This process is for the maintainer of the library only, and is not expected to be followed by other contributers.

* Check out a new release branch from the branch `next-release`.
* Update the version number of the release in `conf.py` for the documentation and `setup.py` for PyPI.
* Update the `changelog.rst` file with the name and date of the release.
* Run the `check.sh` script to run all tests and check they pass.
* Check the documentation is built correctly - check changelog
* Push the release branch to github.
* Remove previously built distributions with `rm dist/*`.
* Build the python package with `python3 setup.py sdist bdist_wheel`.
* Upload the package with `twine upload dist/*`.
* Merge the release branch to the `main` branch.
* Merge the main branch to the `next-release` branch.
