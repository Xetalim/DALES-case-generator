Boundaries and Nesting Cases
============================

These builders cover profile-defined boundaries, three DALES nesting variants,
and cached HARMONIE forecast boundaries. Boundary cases use the output of the
parent run and may require the DALES executable and local input data.

Atmosphere-profile boundaries
-----------------------------

Both cases share a complete base simulation; the second adds multiple time
slices to its boundary profiles:

Required imports:

.. literalinclude:: ../../tests/LBC/test_openbc_atmosphere_profiles.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/LBC/test_openbc_atmosphere_profiles.py
   :language: python
   :pyobject: _build_openbc_atmo_sim

.. literalinclude:: ../../tests/LBC/test_openbc_atmosphere_profiles.py
   :language: python
   :pyobject: openbc_atmosphere_profiles_case

.. literalinclude:: ../../tests/LBC/test_openbc_atmosphere_profiles.py
   :language: python
   :pyobject: openbc_atmosphere_profiles_timedep_case

Triple nesting helpers
----------------------

The three variants below share these grid definitions and case builders:

Required imports:

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: _GRID_SUPER =

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :start-at: _GRID_SUPER =
   :end-at: def _new_grid

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _new_grid

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _attach_common_physics

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _build_outer_parent_sim

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _build_middle_nested_sim

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _build_inner_nested_sim

Triple nesting
--------------

This variant preprocesses and runs the outer and middle simulations before
returning the fine-grid simulation:

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: openbc_triple_nesting_case

Serial triple nesting
---------------------

This variant sets the parent runs to use serial NetCDF statistics:

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: openbc_triple_nesting_case_noparallel

Scaled 64-cell triple nesting
-----------------------------

The scaled case uses a 64 by 64 horizontal grid at each level, shrinking the
domain by a factor of four between nests:

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :start-at: _GRID64_SUPER =
   :end-at: def _build_outer_parent_sim_scaled64

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _build_outer_parent_sim_scaled64

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _build_middle_nested_sim_scaled64

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: _build_inner_nested_sim_scaled64

.. literalinclude:: ../../tests/LBC/test_openbc_triple_nesting.py
   :language: python
   :pyobject: openbc_triple_nesting_scaled64_case

Cached HARMONIE boundaries
--------------------------

These cases read the cached test archive rather than downloading a forecast.
They require the archive and ``cdo``; the timed case also uses a HARMONIE
atmosphere source.

Required imports:

.. literalinclude:: ../../tests/LBC/test_openbc_harmonie_cached.py
   :language: python
   :start-at: import shutil
   :end-before: def extract_zip

.. literalinclude:: ../../tests/LBC/test_openbc_harmonie_cached.py
   :language: python
   :pyobject: extract_zip

.. literalinclude:: ../../tests/LBC/test_openbc_harmonie_cached.py
   :language: python
   :pyobject: openbc_harmonie_cached_case

.. literalinclude:: ../../tests/LBC/test_openbc_harmonie_cached.py
   :language: python
   :pyobject: openbc_harmonie_cached_case_timedep