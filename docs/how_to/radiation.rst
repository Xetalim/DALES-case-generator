Configure radiation
===================

Add ``RadiationModule`` to a simulation that already has a surface module.
Radiation is optional for preprocessing, but a selected radiation scheme may
require scheme-specific runtime data and DALES build capabilities.

.. code-block:: python

   from moda.Radiation import RadiationModule

   radiation = RadiationModule(iradiation=4)
   sim.add(radiation)
   sim.sim_preprocessing_pipeline()

``iradiation`` selects the DALES radiation scheme. Supported values documented
by the module are 0 (off), 2 (parameterized), 3 (surface-LSM radiation), 4
(RRTMG), 5 (RTE-RRTMGP), and 10 (user-specified); value 1 is rejected. The
``timerad`` interval defaults to 60 seconds. When radiation is enabled, the
surface module must provide ``albedoav``. The radiation module writes the
namelist settings and registers the scheme's required input files during
preprocessing.

For a pressure-level or interpolated back-radiation profile, use
``BackradPressureProfile`` or ``BackradInterpolatedProfile`` via the
``backrad_profile`` or ``backrad_interpolated_profile`` option. Input units
and coordinate requirements are documented by those profile constructors in
the :doc:`Radiation API <../api/radiation>`.

The tutorial's constant-temperature surface sets ``albedoav`` and needs no
radiation-specific external dataset. Before running a case with RRTMG or
RTE-RRTMGP, check that the DALES build includes the selected scheme and that
the machine's external-data paths resolve its runtime files.