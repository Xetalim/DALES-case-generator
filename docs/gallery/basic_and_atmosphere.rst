Basic and Atmosphere Cases
==========================

These complete builders cover basic module assembly, timed atmospheric
profiles, and a simulation with multiple physical modules.

Basic simulation
----------------

The smallest full builder in the parameterized run suite:

Required imports:

.. literalinclude:: ../../tests/sim_builders/test_basic.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/sim_builders/test_basic.py
   :language: python
   :pyobject: _build_basic_sim

Time-dependent atmospheric profiles
-----------------------------------

This case adds timed pressure-gradient and surface-temperature forcings:

Required imports:

.. literalinclude:: ../../tests/Atmosphere/test_timedep_atmosphere.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Atmosphere/test_timedep_atmosphere.py
   :language: python
   :pyobject: timedep_atmosphere_case

Time-dependent scalar parameters
--------------------------------

This case places time-dependent values inside profile and surface parameters:

Required imports:

.. literalinclude:: ../../tests/Atmosphere/test_timedep_atmosphere_params.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Atmosphere/test_timedep_atmosphere_params.py
   :language: python
   :pyobject: timedep_atmosphere_with_params_case

LSM, radiation, and serialization setup
----------------------------------------

This case combines atmosphere profiles, a DEPAC land-use setup, uniform soil
state, and radiation:

Required imports:

.. literalinclude:: ../../tests/sim_builders/test_sim2.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/sim_builders/test_sim2.py
   :language: python
   :pyobject: sim_2_case