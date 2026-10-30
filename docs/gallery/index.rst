Gallery
=======

Each example is a complete, runnable simulation intended for adaptation. Both
can generate inputs without external datasets or a DALES binary.

Build your first case
---------------------

Follow the :doc:`end-to-end tutorial <../quickstart>` for installation,
configuration, preprocessing, generated files, and inspection steps.

Time-dependent surface temperature
----------------------------------

This variation supplies surface temperature at 0, 30, and 60 seconds. It
generates a forcing NetCDF file during preprocessing.

The full script, including imports and output configuration, is
`examples/time_dependent_case.py <../examples/time_dependent_case.py>`_. Run it
from the repository root:

.. code-block:: console

   python docs/examples/time_dependent_case.py --output-root generated_cases

For advanced scenarios and test-specific data requirements, see the
:doc:`developer test catalog <../developer/test_catalog>`.