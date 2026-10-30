Radiation and Runtime Cases
===========================

The runtime cases share a base grid, atmosphere, and simulation clock. The
individual builders below add a surface, radiation, LSM, or spraying setup.

Shared runtime setup
--------------------

Required imports:

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: _base_runtime_case

Radiation type 4
----------------

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: radiation_type_4_case

Radiation type 5
----------------

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: radiation_type_5_case

Pressure-level back-radiation profile
-------------------------------------

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: radiation_backrad_profile_case

Interpolated back-radiation profile
-----------------------------------

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: radiation_backrad_interpolated_case

Homogeneous LSM
---------------

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: lsm_homogeneous_runtime_case

Sea-salt spraying
-----------------

.. literalinclude:: ../../tests/sim_builders/test_runtime_special_cases.py
   :language: python
   :pyobject: spraying_runtime_case

KNMI download workflow
----------------------

This builder is part of the parametrized suite but currently skips itself as
too slow. If enabled, it requires KNMI credentials and the ``cdo`` executable.

Required imports:

.. literalinclude:: ../../tests/Atmosphere/test_knmi_download_workflow.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Atmosphere/test_knmi_download_workflow.py
   :language: python
   :pyobject: knmi_download_workflow_case