Surface and Source Cases
========================

The following complete builders exercise IBM, LSM, SLUrb, and emissions.

Immersed boundary method
------------------------

Required imports:

.. literalinclude:: ../../tests/IBM/test_ibm.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/IBM/test_ibm.py
   :language: python
   :pyobject: IBM_case

Land surface model
------------------

Required imports:

.. literalinclude:: ../../tests/Surface/test_lsm.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Surface/test_lsm.py
   :language: python
   :pyobject: simple_LSM_case

HARMONIE soil fields
--------------------

This builder reads a local soil NetCDF file. Replace its test-specific path
with the soil dataset available to your run:

Required imports:

.. literalinclude:: ../../tests/Surface/test_lsm_harmonie_soil.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Surface/test_lsm_harmonie_soil.py
   :language: python
   :pyobject: LSM_with_HARM_soil

SLUrb with IFS/LCZ tiles
------------------------

Required imports:

.. literalinclude:: ../../tests/Surface/test_slurb_lcz.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Surface/test_slurb_lcz.py
   :language: python
   :pyobject: SLURB_LCZ_case

SLUrb with DEPAC tiles
----------------------

The DEPAC case uses the local LCZ builder from the same test module:

Required imports:

.. literalinclude:: ../../tests/Surface/test_slurb_depac.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Surface/test_slurb_depac.py
   :language: python
   :pyobject: SLURB_LCZ_case

.. literalinclude:: ../../tests/Surface/test_slurb_depac.py
   :language: python
   :pyobject: SLURB_DEPAC_case

Basic SLUrb setup
-----------------

Required imports:

.. literalinclude:: ../../tests/Surface/test_slurb.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Surface/test_slurb.py
   :language: python
   :pyobject: simple_SLURB_case

Emission point source
---------------------

Required imports:

.. literalinclude:: ../../tests/Emission/test_emission.py
   :language: python
   :start-after: from __future__ import annotations
   :end-before: def 

.. literalinclude:: ../../tests/Emission/test_emission.py
   :language: python
   :pyobject: emission_case