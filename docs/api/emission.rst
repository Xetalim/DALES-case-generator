Emission
========

Emission sources and tracer configuration exported by ``moda.Emission``.
Add an ``EmissionModule`` to a simulation when emission tracers or point
sources are required; the module writes tracer/source configuration used by
DALES. ``EmissionPointSource`` locations use horizontal grid indices, while
height is in metres, temperature in K, volume in m3, and emission in the
configured tracer's unit per second. The
:doc:`developer/test_catalog <../developer/test_catalog>` links to a complete
test case; the public constructors are listed below.

.. automodule:: moda.Emission
   :members: EmissionModule, EmissionPointSource, EmissionTracer
   :imported-members: