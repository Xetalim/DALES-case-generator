Land surface model
==================

The LSM module, land-use catalogue, and source namespace exported by
``moda.Surface.LSM``. See the :doc:`LSM how-to <../how_to/lsm>` for the
dataset-backed workflow and :doc:`source-priority concept
<../concepts/sources_and_priorities>` for overlay ordering.

``LSMModule`` requires a ``land_use_set`` and values for ``ps`` (Pa),
``z0mav``/``z0hav`` (m), ``iinterp_t``/``iinterp_theta`` (integers 1-4),
and ``dz_soil`` (layer thicknesses in m; list length must equal
``kmax_soil``). ``kmax_soil`` defaults to 4, and ``dz_soil`` defaults to
``[1.89, 0.72, 0.21, 0.07]`` m. ``albedoav`` is dimensionless and becomes
required for radiation schemes that use it. LSM selects ``isurf=11`` and
writes gridded land-surface input under ``input/``; source-specific datasets
may add more files and job transfers.

``include_slurb`` defaults to ``False``. SLURB morphology and namelist settings
are only generated when enabled; SLURB sources and temperature settings must
be compatible with the selected land-use set.

.. automodule:: moda.Surface.LSM
   :members: LSMHomogeneousModule, LSMModule, LandUseSet, lsm_sources
   :imported-members:

.. toctree::
   :maxdepth: 2

   lsm_sources