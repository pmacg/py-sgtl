# Contributing to the SGTL

The next release of the library is tracked by the `git` branch `next-release`.

## Developing an enhancement or bugfix
Thanks for your interest in contributing to this project. Please follow the following process when contributing.

1. Ensure there is an issue on the Github project for the development you would like to do. Feel free to create an issue describing the change you will make.
2. Fork the project.
3. Check out a new branch for the enhancement or bugfix **from the branch `next-release`**.
4. Write tests for the new functionality you are developing or the bug you are fixing.
5. Develop the code for the new functionality or bugfix, committing the code as you go along.
6. Make sure the new code is well documented - full docstrings and comments in the code.
7. Add an entry to the `changelog.rst` file in the documentation folder.
  * If there is no section for the next release yet, add a new release section named 'Unreleased'.
8. Pull and merge the latest `next-release` branch into your development branch to pick up any changes.
9. Run the `check.sh` script to run all of the python tests and to build the documentation, ensure this script succeeds.
10. Check the generated documentation for the new development.
  * This will be in `docs/build/html` after running `check.sh`.
11. Check once again that everything is commited, including the newly generated documentation files, if any.
12. Submit a pull request from your development branch into the `next-releases` branch of the main project.
  * Make sure the pull request is marked as ready to be merged. If not, pull and merge the `next-releases` branch into your development branch again.

## Releasing a new version of the library
This process is for the maintainer of the library only, and is not expected to be followed by other contributers.

1. Check out a new release branch from the branch `next-release`.
2. Update the version number of the release in `conf.py` for the documentation and `setup.py` for PyPI.
3. Update the `changelog.rst` file with the name and date of the release.
4. Run the `check.sh` script to run all tests and check they pass.
5. Check the documentation is built correctly - check changelog
6. Push the release branch to github.
7. Remove previously built distributions with `rm dist/*`.
8. Build the python package with `python3 setup.py sdist bdist_wheel`.
9. Upload the package with `twine upload dist/*`.
10. Merge the release branch to the `main` branch.
11. Merge the main branch to the `next-release` branch.
12. Create a new release in Github named `Version X.X.X` with a new tag `vX.X.X` pointing at the release branch.
13. Delete the release branch.
