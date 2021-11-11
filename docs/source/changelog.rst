Changelog
=========

Unreleased
----------

Fixed
~~~~~
* The `ssbm` method will not allow non-float probabilities

0.3 - 2021-11-11
----------------

Changed
~~~~~~~
* Changed the name of the `sbm` module to `random`.

Added
~~~~~
* Generate standard graphs: complete, cycle, path, star.
* Generate Erdos-Renyi graphs

Fixed
~~~~~
* Spectral clustering checks that number of clusters and eigenvectors are positive integers.

0.2 - 2021-11-10
----------------

Added
~~~~~
* Changelog page to documentation
* ``sbm`` module for generating graphs from the stochastic block model

0.1.1 - 2021-11-10
------------------

Added
~~~~~~
* ``Graph`` class for representing graphs
* ``clustering.spectral_clustering`` clustering method
