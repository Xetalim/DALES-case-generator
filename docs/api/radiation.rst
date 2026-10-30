Radiation
=========

Radiation modules and profile helpers exported by ``moda.Radiation``.
See :doc:`Configure radiation <../how_to/radiation>` for scheme selection,
surface requirements, and runtime prerequisites.

``RadiationModule.iradiation`` accepts 0 (off), 2 (parameterized), 3
(surface-LSM), 4 (RRTMG), 5 (RTE-RRTMGP), or 10 (user specified); 1 is
unsupported. ``timerad`` defaults to 60 seconds. If radiation is enabled,
the attached surface module must provide ``albedoav``. During preprocessing,
the module writes the relevant namelist settings and registers required
radiation files for copying into ``input/`` and the run directory.

.. automodule:: moda.Radiation
   :members: BackradInterpolatedProfile, BackradPressureProfile, NoRadiationModule, ParameterizedRadiationModule, RRTMGRadiationModule, RteRrtmgpRadiationModule, SurfaceLSMRadiationModule, UserRadiationModule
   :imported-members: