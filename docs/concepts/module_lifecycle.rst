Modules and preprocessing
=========================

Simulation modules own a part of a case: a grid, atmospheric state, surface
boundary condition, radiation setup, or run configuration. Construct each
module as a configuration object, then attach it to one ``dales_simulation``
with ``sim.add(module)``. Modules are detached until attached and cannot be
moved to another simulation.

Required pieces
---------------

Before preprocessing, the simulation requires a default namelist, a
``GridDales``, an ``AtmosphereModule``, and a ``SurfaceModule``. The atmosphere
and surface modules mark their presence when attached. Radiation, IBM, LSM,
emissions, and open boundaries are optional unless the experiment requires
them. Some optional modules introduce additional dependencies; for example,
``RadiationModule`` requires a surface module and an albedo when radiation is
enabled.

Pipeline phases
---------------

``sim.sim_preprocessing_pipeline()`` performs these phases in order:

1. Creates the output directory and ``input/``.
2. Configures module-owned namelist settings.
3. Checks required modules and module-specific settings.
4. Prepares module data, including forcing alignment and source application.
5. Writes module-generated NetCDF and other input files.
6. Writes ``job.001``, the namelist, module documentation, and serialized YAML.

The pipeline generates files; it does not invoke ``job.001`` or run DALES.
This separation lets a user generate and inspect a case on a workstation, then
transfer it to a machine with the DALES executable and scheduler.

Output inspection and error diagnosis are part of the workflow; see
:doc:`Build your first case <../quickstart>`. Module constructors and public
contracts are in the :doc:`API reference <../api/index>`.