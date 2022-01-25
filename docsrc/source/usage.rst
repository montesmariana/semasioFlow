Usage
=====

Installation
------------

To use semasioflow, clone the `repository`_ and append it to the path in your Python scripts:

.. code-block:: python

    import os
    os.path.append("/path/to/repository")
    import semasioFlow

.. _repository: https://github.com/montesmariana/semasioFlow/

The "/path/to/repository" is the path leading to the cloned repository, which by default will be called
`semasioFlow`. It will have a subdirectory `semasioFlow` inside, which has the code per se.

Citation
--------

If you use this code, use the information below:

.. .. literalinclude:: ../../CITATION.cff
..    :language: cff

.. code-block:: bibtex

    @software{semasioFlow,
        author = {{Mariana Montes}},
        license = {GPL-3.0-or-later},
        title = {{semasioFlow}},
        version = {0.1.0}
    }