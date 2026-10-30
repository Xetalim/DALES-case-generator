Surface
=======

Surface-flux modules exported by ``moda.Surface``. Its land-surface model
namespace is documented separately in :doc:`Surface LSM <surface_lsm>`.
The :doc:`first-case tutorial <../quickstart>` uses
``ConstantSurfaceTemperatureModule``.

``ConstantSurfaceTemperatureModule`` selects ``isurf=2`` and requires
``thls`` (surface liquid-water potential temperature, K), ``z0mav`` and
``z0hav`` (roughness lengths, m), and ``ps`` (surface pressure, Pa).
``albedoav`` is dimensionless and optional for a non-radiative case; add it
when radiation is enabled. The module writes the ``NAMSURFACE`` settings. The
other constant-flux variants select ``isurf=3`` or ``isurf=4`` and require
their corresponding flux inputs; see their class descriptions below.

For gridded land cover and soil state, use :doc:`LSM <surface_lsm>` rather than
a constant surface boundary condition.

.. automodule:: moda.Surface
   :members: ConstantFluxesModule, ConstantFluxesWithShearModule, ConstantSurfaceTemperatureModule, LSM, SurfaceModule
   :imported-members:

.. toctree::
   :maxdepth: 2

   surface_lsm