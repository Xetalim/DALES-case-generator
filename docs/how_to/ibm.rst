Add immersed-boundary terrain
=============================

This guide starts from a complete simulation with a grid and a surface module.
Add an IBM module before calling preprocessing. The module writes the IBM
height field and associated namelist settings.

.. code-block:: python

   from moda.Geometry.masks import RectangleRealMask
   from moda.IBM import FromGlobalDEM, IBMModification, IBMModule

   ibm = IBMModule(thlibm=293.15, thlroof=293.15)
   ibm.add(FromGlobalDEM("terrain.tif"), priority=10)
   ibm.add(
       IBMModification(
           mask=RectangleRealMask(minx=400, maxx=500, miny=300, maxy=450),
           height=18.0,
       ),
       priority=20,
   )
   sim.add(ibm)
   sim.sim_preprocessing_pipeline()

``thlibm`` and ``thlroof`` are potential temperatures in K and are required.
The raster path must point to a readable georeferenced terrain file. Real-space
mask coordinates are in metres in the grid coordinate system; index masks use
grid indices instead.

Sources run from lower to higher priority; equal priorities preserve insertion
order. Terrain sources overlay positive cells and leave existing values where
they contain zero or negative heights. ``IBMModification`` supports replace,
add, and subtract operations; subtraction clamps heights at zero. Preprocessing
writes the two-dimensional ``bc_height`` field to the IBM NetCDF input and adds
the file transfer to the generated job script.

For AHN inputs, postprocessing, and source grouping, see
:doc:`Working with IBM, LSM, and nested domains <../working_with_ibm_lsm_nesting>`
and the :doc:`IBM API <../api/ibm>`.