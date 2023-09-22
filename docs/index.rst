.. dist-utils documentation master file, created by
   sphinx-quickstart on Fri Sep 22 07:05:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dist-utils's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Module dist_utils
-----------------

Before using ``dist_utils`` functions, you must either call :func:`dist_utils.init_distributed()` or initialize the default process group yourself. :func:`dist_utils.init_distributed()` can be called even if you did not start the script with ``torchrun``: if you did not, it will assume it is the only process and create a process group with a single member.

.. automodule:: dist_utils
   :members:
   :undoc-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
