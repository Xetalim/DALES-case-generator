Building cases
==============

Simulation modules
------------------

``dales_simulation`` is the case container and preprocessing entry point.
Modules are ordinary configuration objects: construct them, configure their
sources, and attach them to a simulation with ``sim.add(module)`` or
``sim += module``. The preprocessing pipeline calls the attached modules to
write their namelist sections and input files.

Keep one module responsible for each part of the case. A typical setup has a
grid, an atmosphere, a surface or land-surface model, radiation, run timing,
and output configuration. The exact combination depends on the DALES run; the
API reference lists the constructors exported by each namespace.

Data sources and ordering
-------------------------

IBM and LSM modules accept source objects through ``add(source, priority=...)``.
The module applies sources from lower to higher priority, so a later source can
replace values written earlier. Equal priorities retain insertion order. Use
this ordering to make overlays explicit and reproducible. Source instances are
configured independently of the simulation and are evaluated when the module
prepares its input.

External datasets and rasters are resolved during preprocessing. Keep their
paths available on the machine generating the case, and check the relevant
module documentation for coordinate systems, supported land-use sets, and
fields that the source writes.

Preparing and running
---------------------

Call ``sim.sim_preprocessing_pipeline()`` after the case is assembled. Review
the generated namelist and job script before execution, especially when using
machine-specific templates or external data. For nested simulations, prepare
and run the parent first so the child can use the parent's boundary and initial
fields; the detailed workflow is covered in
:doc:`Working with IBM, LSM, and nested domains <../working_with_ibm_lsm_nesting>`.