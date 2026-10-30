Core package
============

The main objects exported directly from ``moda``.

Start with :doc:`Build your first case <../quickstart>`. A simulation is
constructed with ``dales_simulation(case_name, machine_conf, timesteps=None)``.
``case_name`` names its output directory; ``machine_conf`` must include
``job_conf.numcores`` and ``case_conf.BASE_OUTPUT_PATH``. The optional
``timesteps`` sequence is in seconds from simulation start and overrides the
union of provider times when time-dependent inputs are present. The pipeline
creates input files but does not run DALES.

Required attachments
--------------------

Before preprocessing, attach a ``DefaultNamelistModule``, ``GridDales``,
``AtmosphereModule``, and a ``SurfaceModule`` subclass. The pipeline raises a
``ValueError`` if any of the first, atmosphere, or surface requirements is
missing; the grid is also needed to configure the domain. Radiation and
specialized modules are optional unless required by the experiment.

``sim.add(module)`` attaches a configured module. Avoid ``sim += module`` in
new code; it remains available but emits a deprecation warning. Attached
modules are bound to that simulation and cannot be moved to another one.

``DefaultNamelistModule`` supplies defaults including ``iexpnr=1``,
``iinput=2``, ``lwarmstart=False``, ``trestart=3600`` seconds, and
``ladaptive=True``. ``TimeModule`` sets calendar and runtime settings:

* ``xtime``: start hour within the day, in hours; default ``0.0``.
* ``xday``: day of year (1-365/366); default ``265``.
* ``xyear``: simulation year; default ``2025``.
* ``runtime``: total run duration in seconds; required, with no default.

Forcing time-axis behavior and its minimum size are described in
:doc:`Forcing time axes <../concepts/forcing_time_axis>`.

.. automodule:: moda
   :members: module_registry, register_module, set_nml_section, setup_logging
   :imported-members:

The root package also re-exports domain namespaces and compatibility names.
Their canonical API descriptions are grouped by domain: grids under
:doc:`Geometry <geometry>`, and the simulation driver, base module, and
time-dependent scalar under :doc:`moda.modular <modular>`.

.. autoclass:: moda.DefaultNamelistModule

.. autoclass:: moda.TimeModule

Output effects
--------------

``sim.sim_preprocessing_pipeline()`` writes ``job.001`` and the namelist under
``input/`` along with a module-annotated namelist Markdown file and serialized
simulation YAML. Modules add their own NetCDF files and required external files
to the input tree. See the tutorial for a concrete file listing and inspection
steps.