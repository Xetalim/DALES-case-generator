Add land-surface data
=====================

This guide assumes ``sim`` already contains the required default namelist,
grid, atmosphere, and surface setup. Replace its simple surface module with an
``LSMModule`` before preprocessing. The LSM needs an explicit land-use
catalogue and required surface/soil settings.

Choose a catalogue
------------------

``LandUseSet`` selects the available tiles and mappings. The example below
uses the IFS catalogue and overlays LCZ land cover, Top10NL fractions, and
BOFEK soil classes. Confirm the selected source supports the catalogue and
that the corresponding files exist under ``spatial_data_path``.

.. code-block:: python

   from pathlib import Path

   from moda.Surface.LSM import LSMModule, LandUseSet
   from moda.Surface.LSM.lsm_sources import FromBofek, FromLCZ, FromTop10

   lsm = LSMModule(
       land_use_set=LandUseSet.IFS,
       ps=100000.0,
       z0mav=0.1,
       z0hav=0.01,
       iinterp_t=1,
       iinterp_theta=1,
       dz_soil=[1.89, 0.72, 0.21, 0.07],
       albedoav=0.22,
   )
   lsm.add(FromLCZ(), priority=100)
   lsm.add(FromTop10(spatial_data_path=Path("data/spatial_data")), priority=200)
   lsm.add(FromBofek(spatial_data_path=Path("data/spatial_data")), priority=300)
   sim.add(lsm)
   sim.sim_preprocessing_pipeline()

``ps`` is surface pressure in Pa; ``z0mav`` and ``z0hav`` are roughness lengths
in m; ``albedoav`` is dimensionless; ``dz_soil`` gives soil layer thicknesses
in m; and the interpolation selectors are DALES integer options in the range
1-4. Higher source priority runs later, so later sources can replace values
written by earlier sources.

The file-backed examples use ``top10nl_landuse_010m.nc`` for ``FromTop10`` and
both ``BOFEK2012_010m.nc`` and ``BOFEK2012_profielen_versie2_1.csv`` for
``FromBofek``. Set ``spatial_data_path`` to the directory containing those
files. ``FromLCZ`` requires a valid grid CRS and obtains its LCZ/ESA inputs
through the LCZ data workflow and its cache; network/data access may be needed
on the first run.

Data and output
---------------

This example needs the LCZ, Top10NL, and BOFEK datasets and a valid grid CRS for
geospatial reprojection. The tutorial does not require these files; the code
above is the point where the external-data workflow begins. Preprocessing
writes the LSM input under ``input/`` and adds any required files to
``job.001``. Inspect the generated NetCDF and the log if a source cannot find
or reproject its data.

For SLURB, enable ``include_slurb=True`` on ``LSMModule`` before adding SLURB
sources. The detailed compatibility and overlay behavior is described in
:doc:`Working with IBM, LSM, and nested domains <../working_with_ibm_lsm_nesting>`.

See :doc:`LSM API <../api/surface_lsm>` and
:doc:`LSM source API <../api/lsm_sources>` for exported constructors.